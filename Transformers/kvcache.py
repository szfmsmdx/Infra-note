from collections import defaultdict

class KVcache():
    def __init__(self):
        """
        cache_dict : {
            'encoder_memory': {
                i : tuple(torch.Tensor, torch.Tensor) -> k_cache, v_cache   # for cross_attn
            },
            'decoder_memory': {
                i : tuple(torch.Tensor, torch.Tensor) -> k_cache, v_cache   # for self_attn
            }
        }
        """
        cache_dict = defaultdict()