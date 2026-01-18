from model.Tokenizer.BPE import BPE_Tokenizer
from model.Config import LlamaConfig
from model.Llama import LlamaGPT

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

def run_profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    tokenizer = BPE_Tokenizer.load("../Transformers/Tokenizer/tokenizer.pt")
    config = LlamaConfig(tokenizer)
    config.num_layers = 16 
    config.dim = 512
    config.num_heads = 16
    config.intermediate_size = 1024
    config.num_kv_heads = 4
    model = LlamaGPT(config).to(device).bfloat16() # 使用 FP16，更符合推理场景
    
    bsz = 4
    seq_len = 256
    input_ids = torch.randint(0, config.vocab_size, (bsz, seq_len)).to(device)

    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            model(input_ids)

    print("Profiling...")
    with profile(
        activities=[
            ProfilerActivity.CPU, 
            ProfilerActivity.CUDA
        ], 
        # llm_profile=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/llama_profile'),
        with_stack=True, # 这个很关键，记录 Python 调用栈
        record_shapes=True,
        profile_memory=True  # 建议开启，可以看到算子的内存分配情况
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model(input_ids)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

if __name__ == "__main__":
    run_profile()