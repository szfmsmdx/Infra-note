from collections import defaultdict

class KVcache():
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.encoder_cache = defaultdict()
        self.decoder_cache = defaultdict()

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