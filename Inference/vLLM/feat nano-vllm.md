>给 nano-vllm 添加一些新特性

# 初步测试
首先我们对 nano-vllm 进行更细粒度的测量，不仅仅是测量他的总吞吐
```json
{
    "timestamp": "2026-02-10T10:54:17.098451",
    "model": "/data3/szf_hf/huggingface/model/Qwen2.5-0.5B",
    "config": {
        "num_seqs": 256,
        "max_input_len": 1024,
        "max_output_len": 1024,
        "tensor_parallel_size": 1,
        "enforce_eager": false,
        "kvcache_block_size": 256
    },
    "metrics": {
        "total_time_s": 6.264,
        "total_tokens_processed": 276793,
        "total_tokens_generated": 133710,
        "throughput_tok_s": 44187.92,
        "prefill_throughput_tok_s": 124745.75,
        "decode_throughput_tok_s": 26168.78,
        "avg_ttft_ms": 728.75,
        "p99_ttft_ms": 1147.84,
        "avg_tpot_ms": 0.08,
        "p99_tpot_ms": 0.75
    }
}
```

我们分别测试了prefill、decode阶段的吞吐以及 ttft 和 tpot 两个指标：
- avg_ttft_ms：从**提交请求**到第一个字符产生的平均时间
- p99_ttft_ms：最慢的1%请求的首字延迟
	- 如果一个超长 prompt 在进行 prefill，那么他会堵住后面的请求——所以我们需要 chunk prefill
- avg_tpot_ms：平均生成一个新token花费的时间
- p99_tpot_ms：最慢1%token的生成平均时间
	- prefill 会抢夺资源，会使得 decode 的耗时猛增，pd 分离让 tpot 变得更加稳定

所以我们考虑做 chunk prefill 和 PD 分离来缓解这个问题
- 一方面，长 prompt 会拉长短 prefill 的时间，所以我们考虑 chunk prefill
- 另一方面，nano-vllm 是 prefill 优先，所以 prefill 可能会抢占已有快结束的 decode，所以 PD 分离能够一定程度上消除“气泡”和抖动
- PD 分离的主要开销是在 KV Cache的迁移上，所以我们最后需要关注新的 ttft、tpot、和网络延迟等指标

# chunk prefill
emmm，实际上我们加入了 chunk prefill 后的效果并不太理想

|**配置 (Config)**|**Prefill 吞吐 (tok/s)**|**平均 TTFT (ms)**|**额外开销 (Overhead)**|
|---|---|---|---|
|**Baseline (无 Chunk)**|**124,745 (基准)**|**728.75 (最优)**|0%|
|**Chunk 512**|122,152 (↓2%)|768.23 (+5%)|低|
|**Chunk 256**|121,111 (↓3%)|806.46 (+10%)|中|
|**Chunk 128**|117,742 (↓5.6%)|880.02 (+20%)|高|
|**Chunk 64**|118,161 (↓5.3%)|939.69 (+29%)|很高|
|**Chunk 1024 (异常)**|108,780 (↓12%)|880.00 (+20%)|**值得注意**|

