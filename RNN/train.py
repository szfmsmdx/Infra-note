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

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    num_epochs = 300

    for epoch in range(num_epochs):
        x = input_ids[:, :-1]    # x : [B, T-1]
        y = input_ids[:, 1:]    # y : [B, T-1]

        logits = model(x)       # logits : [B, T-1, vocab_size]
        B, T, V =  logits.shape
        loss = criterion(
            logits.reshape(B * T, V), y.reshape(B * T) 
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == -1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # 验证推理
    model.eval()
    with torch.no_grad():
        test_text = "我喜欢"
        test_input = tokenizer.encode([test_text]).to(device)
        gen_idx = model.generate(test_input, max_new_token=10)
        gen_text = tokenizer.decode(gen_idx)

        print(f"Prompt: {test_text}")
        print("Generate:", gen_text)


if __name__ == "__main__":
    train()
