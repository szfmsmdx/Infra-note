class T5Config:
    def __init__(self, tokenizer=None, **kwargs):
        # 基础架构参数
        self.vocab_size = kwargs.get("vocab_size", 32128)
        self.model_dim = kwargs.get("model_dim", 512)
        self.ffn_dim = kwargs.get("ffn_dim", 2048)
        self.num_layers = kwargs.get("num_layers", 4)
        self.num_head = kwargs.get("num_head", 8)
        self.dropout_rate = kwargs.get("dropout_rate", 0.1)
        self.max_length = kwargs.get("max_length", 512)

        if tokenizer:
            self.pad_token_id = tokenizer.special_tokens.get("<pad>", 0)
            self.eos_token_id = tokenizer.special_tokens.get("<eos>", 1)
            self.decoder_start_token_id = tokenizer.special_tokens.get("<bos>", 2)
            self.vocab_size = len(tokenizer.id2token)

class LlamaConfig:
    def __init__(self, tokenizer=None, **kwargs):
        self.vocab_size = 32000      
        self.dim = 512               
        self.intermediate_size = 1376 
        self.num_layers = 4          
        self.num_heads = 8           
        self.pad_token_id = 0

        if tokenizer:
            self.pad_token_id = tokenizer.special_tokens.get("<pad>", 0)
            self.eos_token_id = tokenizer.special_tokens.get("<eos>", 1)
            self.decoder_start_token_id = tokenizer.special_tokens.get("<bos>", 2)
            self.vocab_size = len(tokenizer.id2token)