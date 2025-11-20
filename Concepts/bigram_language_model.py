import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Parameters
TEXT_DATA_PATH = "tiny_shiekspear.txt"
VOCAB_SIZE = 65
TRAIN_SPLIT = 0.9
BLOCK_SIZE = 8
BATCH_SIZE = 16
MAX_NEW_TOKENS = 1000
EPOCHS = 30000
EVAL_ITERS = 100
EVAL_INTERVAL = 250
LEARNING_RATE = 1e-3
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
train_data = train_data_tensor[: int(len(train_data_tensor) * 0.9)]
val_data = train_data_tensor[int(len(train_data_tensor) * 0.9) :]


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embedding_lookup_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        x = self.embedding_lookup_table(idx)
        if target is None:
            return x, None
        return x, F.cross_entropy(x.transpose(-2, -1), target)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


writer = SummaryWriter()
model = BigramLanguageModel(VOCAB_SIZE).to(DEVICE)
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

model = model.to("cpu")
writer.flush()
writer.close()
print(
    decode(
        model.generate(torch.ones(1, 1, dtype=torch.long), MAX_NEW_TOKENS)[0].tolist()
    )
)
