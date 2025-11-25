import torch
from torch import nn
import torch.nn.functional as F
import time

# Parameters
TEXT_DATA_PATH = "tiny_shiekspear.txt"
VOCAB_SIZE = 65
TRAIN_SPLIT = 0.9
BLOCK_SIZE = 64
BATCH_SIZE = 4
MAX_NEW_TOKENS = 1000
EPOCHS = 10000
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")
with open(TEXT_DATA_PATH, "r") as file:
    text_data = file.read()

chars = sorted(list(set(text_data)))

# Bigram payers
char_pairs = [a + b for a in chars for b in chars]

stoi = {ch: i for i, ch in enumerate(char_pairs)}
itos = {i: ch for i, ch in enumerate(char_pairs)}


# Update encoder to encode pairs of characters
def encode(text):
    if (
        len(text) % 2
    ):  # Since we only have character pairs, handle situation with odd number of input characters
        text = text + "\n"
    return [stoi[text[idx : idx + 2]] for idx in range(0, len(text), 2)]


def decode(indices):
    return "".join([itos[i] for i in indices])


def get_batch(data, batch_size, block_size):
    indices = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    X = torch.stack([data[idx : idx + block_size] for idx in indices])
    y = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in indices])
    return X, y


train_data_tensor = torch.tensor(encode(text_data), dtype=torch.long).to(DEVICE)
train_data = train_data_tensor[: int(len(train_data_tensor) * 0.9)]
val_data = train_data_tensor[int(len(train_data_tensor) * 0.9) :]


class TrigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embedding_lookup_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        x = self.embedding_lookup_table(idx)
        if target is None:
            return x, None
        return x, F.cross_entropy(x.transpose(-2, -1), target)

    def generate(self, idx, max_new_tokens):  # idx is the start seed token
        # idx is [B, T], ideally where T=1, so a batch of input seed tokens
        for _ in range(max_new_tokens):
            logits, _ = self(
                idx
            )  # Generate logits for what should come next for each B
            logits = logits[:, -1, :]
            probs = F.softmax(
                logits, dim=1
            )  # We softmax the logits to convert them into probabilities
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # We pick the next token from a distribution
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # We concatenate the newly generated token to the end of the starting sequence
        return idx


model = TrigramLanguageModel(len(char_pairs)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start_time = time.time()
for steps in range(EPOCHS):
    xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Total Time Taken: {time.time() - start_time}")

print(loss.item())

model = model.to(torch.device("cpu"))

print(
    decode(
        model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=100)[
            0
        ].tolist()
    )
)
