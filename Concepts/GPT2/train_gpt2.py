from dataclasses import dataclass
import time
import math
import torch
import tiktoken
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Dict, Optional

# Optimization Notes:
# This code is being run on an RTX 3060... Yeah.
# Default takes an average of 8500s, 1880tps
# Optimization 1: Instead of using fp32, use tf32 for floats (7700s, 2100tps)
# Optimization 2: Use BFloat16 (7800s, 2000tps) # Not a significant, most likely because of the GPU I am using.
# Optimization 3: Use Autcast with bfloat16.... but I can't do that becaue my GPU doesn't support this :( (Apparently windows + triton-windows + consumer gpu = bugs)
# Optimization 4: torch.compile ... lead to (42000 ms, 400tps), something went terribly wrong (windows issue. Can't use triton on windows, thus no compilation on windows)
# Optimization 5: Flash Attention ... Only works on Ubuntu because fuck the corporate slave
# Optimization 6: Ugly numbers -> Pretty numbers. Changing vocab size from 50257 to 50304 (divisible by 32), counter-intuitively gives better tps... huh.
# Optimization 7: In AdamW, use fused=True
# Optimization 8: Distributed Data Parallel to use multiple gpus by running identical copies of the same model on a disjoint set of the data using multiple GPUs

# Improvement Notes:
# Improvement 1: Gradient Clipping
# Improvement 2: LR scheduling
# Improvement 3: AdamW weight decay

# Gradient Accumulation
# My GPU can only fit 16 batches of 1024, but GPT-3 used 0.5 M batch ON THE LOWER END... WTF
# So to simulate sequentially any arbitrary batch size, we use gradient accumulations


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # (B, T, C) batch_size, sequence_length, embedding_channels

        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)  # B, T,

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.proj = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.proj(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        # The mean of 0 and std of 0.02 comes directly from the GPT2 paper
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class DataLoader:
    def __init__(self, batch_size, context_length):
        self.batch_size = batch_size
        self.context_lenth = context_length

        with open(TEXT_DATA_PATH, "r") as file:
            text_data = file.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text_data)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} Tokens")
        print(f"1 epoch =  {len(self.tokens) // (batch_size*context_length)} Batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.batch_size, self.context_lenth

        buf = self.tokens[self.current_position : self.current_position + (B * T) + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


torch.manual_seed(42)
torch.cuda.manual_seed(42)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TEXT_DATA_PATH = "tiny_shiekspear.txt"
LEARNING_RATE = 3e-4
EPOCHS = 10
BATCH_SIZE = 8
CONTEXT_WINDOW = 1024
total_batch_size = 524288  # (512 x 1024)
assert total_batch_size % (BATCH_SIZE * CONTEXT_WINDOW) == 0
grad_accum_steps = total_batch_size // (BATCH_SIZE * CONTEXT_WINDOW)

print(f"Total Desired batch size: {total_batch_size}")
print(f"Calculated Gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(BATCH_SIZE, CONTEXT_WINDOW)

# torch.set_float32_matmul_precision("high")  # optimization 1
torch.set_float32_matmul_precision("medium")  # optimization 2


model = GPT(GPTConfig(vocab_size=50304))  # optimization 6
model.to(DEVICE)

# model = torch.compile(model)  # optimization 4

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=True)
for i in range(EPOCHS):
    t0 = time.time()

    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        logits, loss = model(x.to(DEVICE), y.to(DEVICE))
        loss = (
            loss / grad_accum_steps
        )  # loss is cross_entropy, which uses a mean reduction. To keep the gradients from exploding, we need this "normalizer"
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient Clipping
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000  # Time diff in miliseconds
    token_processed = (
        train_loader.batch_size * train_loader.context_lenth * grad_accum_steps
    )
    tokens_per_sec = token_processed / dt
    print(
        f"step {i}, loss: {loss_accum.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}"
    )  # Adding time control before starting optimizations


# Eval
num_return_sequences = 5
max_length = 30
model.eval()

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello! I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = torch.unsqueeze(tokens, 0).repeat(num_return_sequences, 1)
x = tokens.to(DEVICE)

while x.size(1) < max_length:
    with torch.no_grad():
        # optimization 3
        # with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        #     logits, loss = model(x)
        #     import code; code.interact(local=locals())
        logits, loss = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_inidces = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_inidces, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
