import sys
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial  # 用于给 collate_fn 传参

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Tokenizer.BPE import BPE_Tokenizer


def t5_span_corruption(input_ids: list[int], tokenizer: BPE_Tokenizer, noisy_density=0.05):
    """
    返回:
        source_ids: list[int] (带哨兵的输入)
        target_ids: list[int] (哨兵+原始内容+EOS)
    """
    length = len(input_ids)
    num_noisy_token = int(length * noisy_density)
    num_noisy_spans = max(1, int(num_noisy_token / 3))

    mask_indices = set()
    for _ in range(num_noisy_spans):
        start = random.randint(0, max(0, length - 4))
        span_len = random.randint(1, 4)
        for i in range(start, start + span_len):
            if i < length:
                mask_indices.add(i)

    source_id = []
    target_id = []

    try:
        sentinel_start_id = tokenizer.token2id["<extra_id_0>"]
    except Exception:
        raise ValueError("Tokenizer 中必须先注册 <extra_id_0> 等哨兵 token！")

    sentinel_idx = 0
    is_in_mask = False

    # Source (输入): 把挖掉的地方变成哨兵 <extra_id_0> -> [10, <id_0>, 40, 50]
    # Target (标签): 把挖掉的内容拼在哨兵后面 -> <id_0>, 20, 30, <eos>

    for i, token_id in enumerate(input_ids):
        if i in mask_indices:  # 需要插入
            if not is_in_mask:  # 第一个位置
                sentinel_id = sentinel_start_id + sentinel_idx
                source_id.append(sentinel_id)
                target_id.append(sentinel_id)
                is_in_mask = True
                sentinel_idx += 1
            target_id.append(token_id)
        else:
            is_in_mask = False
            source_id.append(token_id)

    if "<eos>" in tokenizer.token2id:
        target_id.append(tokenizer.token2id["<eos>"])

    return source_id, target_id


class T5Datasets(Dataset):
    def __init__(self, file_list, tokenizer: BPE_Tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        if not isinstance(file_list, list):
            file_list = [file_list]

        for file in file_list:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if len(input_ids) > self.max_length - 10:
            input_ids = input_ids[:self.max_length - 10]

        source_ids, target_ids = t5_span_corruption(input_ids, self.tokenizer, 0.15)
        return {
            "input_ids": torch.tensor(source_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long)  # 注意：你原写的是 "lables"，应为 "labels"
        }

def t5_collect_fn(batch, pad_id=0, decoder_start_id=2):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_pad = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    label_pad = pad_sequence(labels, batch_first=True, padding_value=pad_id)

    batch_size = label_pad.size(0)
    starts = torch.full((batch_size, 1), decoder_start_id, dtype=torch.long)
    decoder_input_ids = torch.cat([starts, label_pad[:, :-1]], dim=1)

    attention_mask = (input_pad != pad_id).long()
    
    return {
        "input_ids": input_pad,
        "decoder_input_ids": decoder_input_ids,
        "labels": label_pad,
        "attention_mask": attention_mask
    }


if __name__ == "__main__":
    
    
    tokenizer_path = os.path.join(project_root, "Tokenizer", "tokenizer.pt")
    tokenizer = BPE_Tokenizer()
    
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer.load(tokenizer_path)
    else:
        print("⚠️ Tokenizer not found, training a dummy one...")
        tokenizer.train(["测试数据用于初始化"])

    print("Adding special tokens...")
    sentinels = [f"<extra_id_{i}>" for i in range(100)]
    tokenizer.add_special_token(sentinels)
    tokenizer.add_special_token(["<pad>", "<eos>", "<unk>"])
    
    pad_id = tokenizer.token2id.get("<pad>", 0)
    print(f"PAD ID: {pad_id}")
    
    test_file = "test_dummy_data.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("短句子。\n") 
        f.write("这是一个稍微长一点的测试句子，用来检查 padding 是否工作正常。\n")
        f.write("非常非常长的句子，如果不截断可能会溢出，但是我们有 max_length 保护。\n")
        
    # --- 3. 实例化 Dataset ---
    dataset = T5Datasets(file_list=test_file, tokenizer=tokenizer, max_length=128)
    print(f"\nDataset size: {len(dataset)}")

    # 使用 partial 把 pad_id 传入 collate_fn
    loader = DataLoader(
        dataset, 
        batch_size=2,     # 设为 2，强迫它把长短不一的句子拼在一起
        shuffle=False, 
        collate_fn=partial(t5_collect_fn, pad_id=pad_id)
    )
    
    print("\n--- Testing DataLoader & Batching ---")
    
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        mask = batch["attention_mask"]
        
        print(f"\n[Batch {batch_idx}]")
        print(f"Input Shape: {input_ids.shape}") # 期望: [2, Max_Len_in_Batch]
        print(f"Labels Shape: {labels.shape}")
        
        print("Input Tensor Content:")
        print(input_ids)
        
        print("Attention Mask Content (Should identify padding):")
        print(mask)
        
        decoded_src = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        decoded_tgt = tokenizer.decode(labels[0].tolist(), skip_special_tokens=False)
        print(f"\nDecoded Sample Source: {decoded_src}")
        print(f"\nDecoded Sample Target: {decoded_tgt}")
        
    # --- 5. 清理 ---
    if os.path.exists(test_file):
        os.remove(test_file)
    print("\n✅ Test Completed.")