import torch
import torch.nn as nn
from torch.optim import Adam

from Tokenizer import Tokenizer, build_vocab
from RNN import RNN_Net

def train():
    sentences = [
        "你好！",
        "今天天气真好。",
        "我喜欢学习。",
        "你好，世界！"
    ]

    vocab = build_vocab(sentences)
    tokenizer = Tokenizer(vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer.encode(sentences)  # [B, T]
    input_ids = input_ids.to(device)

    embed_dim = 16
    hidden_dim = 32

    model = RNN_Net(
        vocab_size=tokenizer.vocab_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        pad_idx=tokenizer.pad_id,
        eos_idx=tokenizer.eos_id,
        device=device
    ).to(input_ids.device)

    optimizer = Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    model.train()
    num_epochs = 50

    for epoch in range(num_epochs):
        x = input_ids[:, :-1]   # [B, T-1]
        y = input_ids[:, 1:]    # [B, T-1]

        logits = model(x)       # [B, T-1, vocab]
        B, T, V = logits.shape

        loss = criterion(
            logits.reshape(B * T, V),
            y.reshape(B * T)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_input = tokenizer.encode(["你好"])
        test_input = test_input.to(input_ids.device)

        gen_ids = model.generate(test_input, max_new_token=10)
        gen_text = tokenizer.decode(gen_ids)

        print("Prompt: 你好")
        print("Generate:", gen_text)


if __name__ == "__main__":
    train()
