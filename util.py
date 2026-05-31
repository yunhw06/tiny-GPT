import torch
from torch.utils.data import Dataset, DataLoader
from  dataset import NextTokenDataset

def load_dataloader(text_path, block_size, batch_size = 64):
    text = open(text_path, "r", encoding="utf-8").read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    vocab_size = len(chars)
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    block_size = block_size # 64
    dataset = NextTokenDataset(data, block_size)

    loader = DataLoader(dataset, batch_size, shuffle=True)

    info = {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "stoi": stoi,
        "itos": itos,
    }

    return loader, info

