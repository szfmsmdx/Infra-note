import torch
import torch.nn.functional as F

def build_vocab(corpus: str):
    tokens = corpus.split()
    vocab_list = sorted(set(tokens)) 
    word2idx = {t: i for i, t in enumerate(vocab_list)}
    idx2word = {i: t for i, t in enumerate(vocab_list)}
    return word2idx, idx2word, len(vocab_list)

def skip_gram(tokens, word2idx, window=2):
    """生成 (center, target) 索引对"""
    #TODO:实现负采样
    data = []
    for i in range(len(tokens)):
        center_token = tokens[i]
        center_idx = word2idx[center_token]

        for j in range(i - window, i + window + 1):
            if j >= 0 and i != j and j < len(tokens):
                data.append((center_idx, word2idx[tokens[j]]))
    return data

def cbow(tokens, word2idx, window=2):
    """生成 ([context_idxs], center_idx)"""
    #TODO:实现负采样
    data = []
    for i in range(len(tokens)):
        center_token = tokens[i]
        center_idx = word2idx[center_token]

        context_idx = []
        for j in range(i - window, i + window + 1):
            if j >= 0 and i != j and j < len(tokens):
                context_idx.append(word2idx[tokens[j]])
        if context_idx:
            data.append((context_idx, center_idx))
    return data

class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.u_embeddings = torch.nn.Embedding(vocab_size, embed_dim, device=device)
        self.v_embeddings = torch.nn.Linear(embed_dim, vocab_size, bias=False, device=device)
        self.device = device

    def forward(self, input, mode="skip_gram"):
        input.to(device)
        input_embed = self.u_embeddings(input)

        if mode == "cbow":
            """这时 input embed : [B, window, dim]"""
            input_embed = torch.mean(input_embed, dim=1)

        score = self.v_embeddings(input_embed)
        return score

def train(model, tokens : list[str], word2idx, window=2, mode="cbow", num_epochs=50):
    print(f" ----------- mode : {mode} -----------")
    if mode == "cbow":
        train_data = cbow(tokens, word2idx, window)
    else:
        train_data = skip_gram(tokens, word2idx, window)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for data_item in train_data:
            x, y = data_item
            if mode == "cbow":
                x_tensor = torch.tensor(x, dtype=torch.long, device=model.device).unsqueeze(0)
            else:
                x_tensor = torch.tensor([x], dtype=torch.long, device=model.device)

            y_tensor = torch.tensor([y], dtype=torch.long, device=model.device)
            optimizer.zero_grad()
            
            pred = model(x_tensor, mode=mode) # Pred shape: [1, Vocab_Size]
            loss = criterion(pred, y_tensor)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_data)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

# --- 4. 推理测试 (新增功能) ---
def get_similar_words(model, word, word2idx, idx2word, top_k=3):
    """
    计算余弦相似度，找出最接近的词
    """
    if word not in word2idx:
        return "Word not in vocab"
    
    word_idx = word2idx[word]
    word_vec = model.u_embeddings.weight.data[word_idx] 
    
    all_weights = model.u_embeddings.weight.data
    
    cos_sim = F.cosine_similarity(word_vec.unsqueeze(0), all_weights)
    
    vals, indices = torch.topk(cos_sim, top_k + 1) # +1 是因为自己和自己最像，要排除
    
    print(f"\n[{word}] is similar to:")
    for i in range(1, len(indices)): # 从 1 开始跳过自己
        idx = indices[i].item()
        sim = vals[i].item()
        print(f"  - {idx2word[idx]} (score: {sim:.3f})")
        
if __name__ == "__main__":
    corpus = """
    king queen prince princess kingdom royal
    king queen prince princess kingdom royal
    apple banana orange fruit juice eat
    apple banana orange fruit juice eat
    computer python code ai data learning
    computer python code ai data learning
    king eat apple
    queen eat orange
    coder write python
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = corpus.split()
    word2idx, idx2word, vocab_size = build_vocab(corpus)
    
    print("\n========== Experiment 1: CBOW Only ==========")
    model_cbow = Word2Vec(vocab_size, embed_dim=8, device=device) # 重新初始化
    model_cbow = train(model_cbow, tokens, word2idx, window=2, mode="cbow", num_epochs=1000)
    
    print("\n[CBOW Inference Results]")
    test_words = ["king", "apple", "python"]
    for w in test_words:
        get_similar_words(model_cbow, w, word2idx, idx2word)

    print("\n========== Experiment 2: Skip-gram Only ==========")
    model_skip = Word2Vec(vocab_size, embed_dim=16, device=device) # 关键步骤：重新初始化！
    model_skip = train(model_skip, tokens, word2idx, window=2, mode="skip_gram", num_epochs=200)
    
    print("\n[Skip-gram Inference Results]")
    for w in test_words:
        get_similar_words(model_skip, w, word2idx, idx2word)