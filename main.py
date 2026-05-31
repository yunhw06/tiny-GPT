import torch
from util import load_dataloader
from inference import sample_gpt
from model import TinyGPT
from train import train

if __name__ == "__main__":
    
    loader, info = load_dataloader("./data/input.txt", block_size=64, batch_size=64)
    xb, yb = next(iter(loader))

    model = TinyGPT(info)
    logits = model(xb)
    print("logits.shape:", logits.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT(info).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model = train(model, loader, optimizer, device, num_epochs=100, max_steps_per_epoch=300)

    print(sample_gpt(model, info, device, start_text="ROMEO:", max_new_tokens=500))



