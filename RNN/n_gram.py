import random
from collections import defaultdict, Counter

class SimpleNGram:
    def __init__(self, n=3):
        self.n = n
        self.model = defaultdict(Counter)

    def train(self, corpus):
        """
        训练过程：简单的滑动窗口计数
        """
        tokens = corpus.split()
        
        # 滑动窗口: [i, i+n]
        for i in range(len(tokens) - self.n + 1):
            # 取出 n 个词
            gram = tokens[i : i + self.n]
            
            # 前 n-1 个词作为 context (历史)
            context = tuple(gram[:-1])
            # 第 n 个词作为 target (预测目标)
            target = gram[-1]
            
            # 统计频率
            self.model[context][target] += 1

    def predict_next(self, context_text, sample=False):
        """
        根据 context 预测下一个词
        sample=True: 按概率采样 (Temperature sampling 的雏形)
        sample=False: 贪婪搜索 (Greedy search, 取概率最大的)
        """
        tokens = context_text.split()
        # 截取最后 n-1 个词作为 context
        context_tuple = tuple(tokens[-(self.n - 1):])
        
        if context_tuple not in self.model:
            return None # OOV 或 Context 未见
            
        candidates = self.model[context_tuple]
        
        if not sample:
            # 贪婪模式：直接返回出现次数最多的词
            return candidates.most_common(1)[0][0]
        else:
            # 采样模式：根据频率分布进行加权随机选择
            words = list(candidates.keys())
            counts = list(candidates.values())
            return random.choices(words, weights=counts, k=1)[0]

    def calculate_prob(self, context_text, target_word):
        """
        计算条件概率 P(target | context)
        """
        tokens = context_text.split()
        context_tuple = tuple(tokens[-(self.n - 1):])
        
        if context_tuple not in self.model:
            return 0.0
            
        candidates = self.model[context_tuple]
        total_count = sum(candidates.values())
        target_count = candidates[target_word]
        
        return target_count / total_count

# --- 测试代码 ---
if __name__ == "__main__":
    # 语料 (模拟一个小语料库)
    text = """
    deep learning is amazing deep learning is fun 
    deep learning is hard machine learning is fun
    deep learning not fun
    """
    
    # 1. 初始化 3-gram 模型 (根据前2个词预测第3个)
    ngram = SimpleNGram(n=3)
    
    # 2. 训练
    ngram.train(text)
    
    # 3. 预测
    ctx = "deep learning"
    next_word = ngram.predict_next(ctx)
    prob = ngram.calculate_prob(ctx, next_word)
    
    print(f"Context: '{ctx}'")
    print(f"Prediction: '{next_word}' (Probability: {prob:.2f})")
    
    # 查看模型内部存储结构 (便于理解)
    print("\nModel Internal State (context -> counts):")
    # deep learning 后面接了 is (2次)
    print(f"('deep', 'learning'): {ngram.model[('deep', 'learning')]}")