import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

if not Path("names.txt").exists():
    !wget -q https://raw.githubusercontent.com/karpathy/makemore/master/names.txt

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
chars = ['.'] + chars

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)
encoded_words = [[stoi[ch] for ch in w] for w in words]

print("num words:", len(words))
print("vocab_size:", vocab_size)
print(words[:10])

class NamesContextDataset(Dataset):
    def __init__(self, encoded_words, block_size):
        self.X, self.Y = [], []
        for word in encoded_words:
            context = [0] * block_size
            for ix in word + [0]:
                self.X.append(context.copy())
                self.Y.append(ix)
                context = context[1:] + [ix]
        self.X = torch.tensor(self.X, dtype=torch.long)
        self.Y = torch.tensor(self.Y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

block_size = 1
dataset = NamesContextDataset(encoded_words, block_size)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

x, y = dataset[1]
xb, yb = next(iter(loader))
print("single x:", x, x.shape)
print("single y:", y)
print("batch xb.shape:", xb.shape)
print("batch yb.shape:", yb.shape)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.W = nn.Parameter(torch.randn(vocab_size, vocab_size))

    def forward(self, x):
        x = x.view(-1)  # (B,)
        x_onehot = F.one_hot(x, num_classes=self.vocab_size).float()  # (B, V)
        logits = x_onehot @ self.W                                    # (B, V)
        return logits

model = BigramLanguageModel(vocab_size)
logits = model(xb)
print("logits.shape:", logits.shape)
print("initial loss:", F.cross_entropy(logits, yb).item())

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)

for epoch in range(20):
    train_loss = train_one_epoch(model, loader, optimizer, device)
    if epoch % 5 == 0 or epoch == 19:
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")

  @torch.no_grad()
def sample(model, block_size, itos, device, num_samples=10, max_len=20):
    model.eval()
    results = []
    for _ in range(num_samples):
        context = torch.zeros((1, block_size), dtype=torch.long, device=device)
        out = []
        for _ in range(max_len):
            logits = model(context)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            next_token = ix.item()
            if next_token == 0:
                break
            out.append(itos[next_token])
            context = torch.cat([context[:, 1:], ix], dim=1)
        results.append("".join(out))
    return results

sample(model, block_size, itos, device, num_samples=10)
