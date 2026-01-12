from collections import defaultdict
import torch


class KVcache():
    def __init__(
            self, 
            num_layers,
            batch_size,
            num_head, 
            seq_len,
            head_dim,
            device
        ):
        """
        self.encoder_cache : {
            idx_layer : (memory_k_cache, memory_v_cache)
        }
        self.decoder_cache : {
            idx_layer : (token_k_cache, token_v_cache)
        }
        """
        self.num_layers = num_layers
        self.encoder_cache = defaultdict(tuple)
        self.decoder_cache = defaultdict(tuple)

        # 预分配 decoder
        for i in range(num_layers):
            self.decoder_cache[i] = (
                torch.zeros((batch_size, num_head, seq_len, head_dim), device=device),  # k
                torch.zeros((batch_size, num_head, seq_len, head_dim), device=device),  # v
            )


    def update_encoder_cache(self, layer_idx, kv_cache):
        if layer_idx >= self.num_layers:
            raise IndexError
        self.encoder_cache[layer_idx] = kv_cache

    def get_encoder_cache(self, layer_idx):
        if layer_idx >= self.num_layers:
            raise IndexError
        return self.encoder_cache[layer_idx]
    
    def update_decoder_cache(self, layer_idx, kv_cache):
        if layer_idx >= self.num_layers:
            raise IndexError
        self.decoder_cache[layer_idx] = kv_cache

    def get_decoder_cache(self, layer_idx):
        if layer_idx >= self.num_layers:
            raise IndexError
        return self.decoder_cache[layer_idx]
    
    def reset(self):
        self.encoder_cache.clear()
        self.decoder_cache.clear()