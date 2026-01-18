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

    def get_encoder_cache(self, layer_idx):
        if layer_idx >= self.num_layers:
            raise IndexError
        return self.encoder_cache[layer_idx]

    def get_decoder_cache(self, layer_idx):
        if layer_idx >= self.num_layers:
            raise IndexError
        return self.decoder_cache[layer_idx]
    
    # def update_encoder_cache(self, layer_idx, kv_cache):
    #     if layer_idx >= self.num_layers:
    #         raise IndexError
    #     self.encoder_cache[layer_idx] = kv_cache

    # def update_decoder_cache(self, layer_idx, kv_cache):
    #     if layer_idx >= self.num_layers:
    #         raise IndexError
    #     self.decoder_cache[layer_idx] = kv_cache
    
    def reset(self):
        self.encoder_cache.clear()
        self.decoder_cache.clear()


class LlamaLayerCache:
    def __init__(self, config, batch_size, device):
        self.max_seq_len = config.max_new_tokens
        self.n_kv_heads = config.num_kv_heads
        self.head_dim = config.dim // config.num_heads
        
        # 预分配连续显存 [B, G, Max_L, D]
        cache_shape = (batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device)
        self.v_cache = torch.zeros(cache_shape, device=device)

    def update(self, k_val, v_val, start_pos):
        """
        k_val/v_val: [B, G, cur_L, D]
        """
        bsz, n_kv_heads, q_len, head_dim = k_val.shape
        
        # 写入当前步的 KV
        self.k_cache[:, :, start_pos : start_pos + q_len, :] = k_val
        self.v_cache[:, :, start_pos : start_pos + q_len, :] = v_val
        
        # 返回 0 到当前位置的所有 KV 视图
        return (
            self.k_cache[:, :, : start_pos + q_len, :],
            self.v_cache[:, :, : start_pos + q_len, :]
        )