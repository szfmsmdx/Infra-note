>参考文章：
>[(26 封私信 / 80 条消息) 【CUDA 入门必知必会】profiling 性能分析 - 知乎](https://zhuanlan.zhihu.com/p/1992746269999928467?share_code=dsKAUN7aZUb2&utm_psn=1996769737401729198)
>

# nsys
```bash
nsys profile --trace=cuda,nvtx --stats=true -o fused_norm_profile python fused_norm_profile.py
```

>nvtx 范围内计时

![[nvtx_sum.png]]
CPU 发起 → GPU 执行 → CPU 等待完成


>所有 cuda api 调用情况以及时间
![[cuda_api_sum.png]]

>kernel 级别调用情况
![[cuda_gpu_kern_sum.png]]

>device - host 访存时间
![[cuda_gpu_mem_time_sum.png]]

> device - host 内存
![[cuda_gpu_mem_size_sum.png]]

# ncu
```bash
sudo ncu \
   --target-processes all \
   --kernel-name regex:fused_add_rms_norm_kernel \
   --launch-skip 10 --launch-count 1 \
   python fused_norm_profile.py

```

![[GPU  speed of light throughput.png]]

![[Launch Statistics.png]]

![[Occupancy.png]]
