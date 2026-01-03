from tqdm import tqdm
import torch
from Config import T5Config
from Transformer import Model
from Dataset.Dataset import T5Datasets
from Tokenizer.BPE import BPE_Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import os
from functools import partial
from Dataset.Dataset import t5_collect_fn

class TrainConfig:
    def __init__(self):
        self.epochs = 25
        self.batch_size = 32
        self.lr = 3e-4
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.save_path = "./checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, train_loader, optimizer, criterion, config:T5Config, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        starts = torch.full((labels.size(0), 1), config.decoder_start_token_id, device=device)
        
        decode_input_ids = torch.cat([starts, labels[:, :-1]], dim=1)
        logits = model(input_ids, decode_input_ids)
        
        loss = criterion(logits.view(-1, config.vocab_size), labels.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)

def train(model, train_dataset, model_config, train_config):
    if not os.path.exists(train_config.save_path):
        os.makedirs(train_config.save_path)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True,
        collate_fn=partial(t5_collect_fn, pad_id=model_config.pad_token_id, decoder_start_id=model_config.decoder_start_token_id)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config.lr, 
        weight_decay=train_config.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model_config.pad_token_id)
    
    model.to(train_config.device)

    for epoch in range(train_config.epochs):
        avg_loss = train_epoch(
            model, train_loader, optimizer, criterion, model_config, train_config.device
        )
        
        print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")
        
        save_file = os.path.join(train_config.save_path, f"t5_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_file)
        print(f"Model saved to {save_file}")

if __name__ == "__main__":
    tokenizer = BPE_Tokenizer()
    tokenizer.load(r".\Tokenizer\tokenizer.pt")
    sentinels = [f"<extra_id_{i}>" for i in range(100)]
    tokenizer.add_special_token(sentinels)
    tokenizer.add_special_token(["<pad>", "<eos>", "<unk>"])
    model_config = T5Config(tokenizer)
    train_config = TrainConfig()
    data_files = [r".\Tokenizer\data\train-cn.txt", r".\Tokenizer\data\train-en.txt"] 
    train_dataset = T5Datasets(file_list=data_files, tokenizer=tokenizer, max_length=model_config.max_length)
    model = Model(model_config)
    print(model)
    train(model, train_dataset, model_config, train_config)