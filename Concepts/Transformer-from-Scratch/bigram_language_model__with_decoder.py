import time

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Parameters
TEXT_DATA_PATH = "tiny_shiekspear.txt"
TRAIN_SPLIT = 0.9
VOCAB_SIZE = 65
BLOCK_SIZE = 256

BATCH_SIZE = 64
N_EMBED = HEAD_SIZE = 384
N_HEADS = 6
N_LAYER = 6

DROPOUT = 0.2
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 500
EVAL_ITERS = 200
EPOCHS = 5000

MAX_NEW_TOKENS = 1000

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

with open(TEXT_DATA_PATH, "r") as file:
    text_data = file.read()

chars = sorted(list(set(text_data)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(text):
    return [stoi[c] for c in text]


def decode(indices):
    return "".join([itos[i] for i in indices])


def get_batch(data, batch_size, block_size):
    indices = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    X = torch.stack([data[idx : idx + block_size] for idx in indices])
    y = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in indices])
    return X, y


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters, train_data, val_data):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = (
                get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
                if split == "train"
                else get_batch(val_data, BATCH_SIZE, BLOCK_SIZE)
            )
            logits, loss = model(X, y)

            losses[k] = loss

        out[split] = losses.mean()
    model.train()
    return out


train_data_tensor = torch.tensor(encode(text_data), dtype=torch.long).to(DEVICE)
train_data = train_data_tensor[: int(len(train_data_tensor) * TRAIN_SPLIT)]
val_data = train_data_tensor[int(len(train_data_tensor) * TRAIN_SPLIT) :]


class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()

        self.n_embed = n_embed

        self.queries = nn.Linear(n_embed, head_size, bias=False)
        self.keys = nn.Linear(n_embed, head_size, bias=False)
        self.values = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        x = wei @ v

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.sa_heads = nn.ModuleList(
            [
                Head(n_embed, head_size // n_heads, block_size, dropout)
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(
            n_embed, n_embed
        )  # Projecting back into the residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.concat([head(x) for head in self.sa_heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Forward(nn.Module):
    def __init__(self, head_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                head_size, 4 * head_size
            ),  # Multiplying by 4 because the transformer paper does say. They have 512 inputs and 2048 hidden dims, so we multiply by 4
            nn.ReLU(),
            nn.Linear(
                4 * head_size, head_size
            ),  # Projecting back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_heads, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.sa_heads = MultiHeadAttention(
            n_heads, n_embed, head_size, block_size, dropout
        )  # Acts as a communication layer between tokens
        self.ffwd = Forward(head_size, dropout)  # Acts as computation within a node
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(
            self.ln1(x)
        )  # Adding residual connections becuase the network is too deep and learning is bad
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(
        self, vocab_size, block_size, n_embed, head_size, n_heads, n_blocks, dropout
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed

        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.positional_embeddings = nn.Embedding(block_size, n_embed)

        self.blocks = nn.Sequential(
            *[
                Block(n_heads, n_embed, head_size, block_size, dropout)
                for _ in range(n_blocks)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        token_emb = self.token_embeddings(x)
        position_emb = self.positional_embeddings(torch.arange(0, T, device=x.device))

        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size :]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


writer = SummaryWriter()
model = BigramLanguageModel(
    VOCAB_SIZE, BLOCK_SIZE, N_EMBED, HEAD_SIZE, N_HEADS, N_LAYER, DROPOUT
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start_time = time.time()
for steps in range(EPOCHS):

    if steps + 1 % EVAL_INTERVAL == 0:
        losses = estimate_loss(model, EVAL_ITERS, train_data, val_data)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        writer.add_scalar("Loss/train", losses["train"], steps + 1)
        writer.add_scalar("Loss/val", losses["val"], steps + 1)

    xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Total Time Taken: {time.time() - start_time}")

print(loss.item())
writer.flush()
writer.close()

print(
    decode(
        model.generate(
            torch.ones(1, 1, dtype=torch.long, device=DEVICE), MAX_NEW_TOKENS
        )[0].tolist()
    )
)
