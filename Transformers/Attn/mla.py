#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


MB = 1024 * 1024


@dataclass
class StageMetrics:
    time_s: float
    peak_bytes: int
    peak_increase_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MLA vs Native Attention (prefill + decode with cache)."
    )
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--gen-lens", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--pair-mode", choices=["zip", "cartesian"], default="zip")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / MB


def stage_benchmark(
    fn: Callable[[], object],
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[StageMetrics, object]:
    last_out = None
    for _ in range(warmup):
        last_out = fn()
    sync_if_cuda(device)

    times: list[float] = []
    peaks: list[int] = []
    peak_increases: list[int] = []

    for _ in range(repeat):
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline = 0

        sync_if_cuda(device)
        start = time.perf_counter()
        last_out = fn()
        sync_if_cuda(device)
        end = time.perf_counter()

        times.append(end - start)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            peaks.append(int(peak))
            peak_increases.append(int(max(0, peak - baseline)))
        else:
            peaks.append(0)
            peak_increases.append(0)

    metric = StageMetrics(
        time_s=float(statistics.mean(times)),
        peak_bytes=max(peaks) if peaks else 0,
        peak_increase_bytes=max(peak_increases) if peak_increases else 0,
    )
    return metric, last_out


class NativeAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bsz, seq_len, self.num_heads * self.head_dim)

    def prefill(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.o_proj(self._merge_heads(y))
        return y, k, v

    def decode_step(
        self,
        x_step: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(x_step))
        k_new = self._split_heads(self.k_proj(x_step))
        v_new = self._split_heads(self.v_proj(x_step))
        k_cache = torch.cat([k_cache, k_new], dim=2)
        v_cache = torch.cat([v_cache, v_new], dim=2)
        y = F.scaled_dot_product_attention(q, k_cache, v_cache, is_causal=False)
        y = self.o_proj(self._merge_heads(y))
        return y, k_cache, v_cache


class MLAAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, latent_dim: int):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.kv_down = nn.Linear(d_model, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, inner_dim, bias=False)
        self.v_up = nn.Linear(latent_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bsz, seq_len, self.num_heads * self.head_dim)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(x))
        c_kv = self.kv_down(x)
        k = self._split_heads(self.k_up(c_kv))
        v = self._split_heads(self.v_up(c_kv))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.o_proj(self._merge_heads(y))
        return y, c_kv

    def decode_step(
        self,
        x_step: torch.Tensor,
        c_kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(x_step))
        c_new = self.kv_down(x_step)
        c_kv_cache = torch.cat([c_kv_cache, c_new], dim=1)
        k = self._split_heads(self.k_up(c_kv_cache))
        v = self._split_heads(self.v_up(c_kv_cache))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = self.o_proj(self._merge_heads(y))
        return y, c_kv_cache


def run_one_config(
    seq_len: int,
    gen_len: int,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    bsz = args.batch_size
    d_model = args.d_model
    num_heads = args.num_heads
    head_dim = args.head_dim
    latent_dim = args.latent_dim
    if num_heads * head_dim <= 0:
        raise ValueError("num_heads * head_dim must be positive.")

    native = NativeAttention(d_model, num_heads, head_dim).to(device=device, dtype=dtype).eval()
    mla = MLAAttention(d_model, num_heads, head_dim, latent_dim).to(device=device, dtype=dtype).eval()

    x_prefill = torch.randn(bsz, seq_len, d_model, device=device, dtype=dtype)
    x_decode = torch.randn(bsz, gen_len, d_model, device=device, dtype=dtype)

    with torch.inference_mode():
        native_prefill_metric, native_prefill_out = stage_benchmark(
            fn=lambda: native.prefill(x_prefill),
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )
        _, native_k_cache, native_v_cache = native_prefill_out
        native_prefill_cache_bytes = (
            native_k_cache.numel() * native_k_cache.element_size()
            + native_v_cache.numel() * native_v_cache.element_size()
        )

        mla_prefill_metric, mla_prefill_out = stage_benchmark(
            fn=lambda: mla.prefill(x_prefill),
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )
        _, mla_c_cache = mla_prefill_out
        mla_prefill_cache_bytes = mla_c_cache.numel() * mla_c_cache.element_size()

        def native_decode_full() -> tuple[torch.Tensor, torch.Tensor]:
            k = native_k_cache.clone()
            v = native_v_cache.clone()
            y = None
            for i in range(gen_len):
                y, k, v = native.decode_step(x_decode[:, i : i + 1, :], k, v)
            assert y is not None
            return k, v

        native_decode_metric, native_decode_out = stage_benchmark(
            fn=native_decode_full,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )
        native_k_final, native_v_final = native_decode_out
        native_final_cache_bytes = (
            native_k_final.numel() * native_k_final.element_size()
            + native_v_final.numel() * native_v_final.element_size()
        )

        def mla_decode_full() -> torch.Tensor:
            c = mla_c_cache.clone()
            y = None
            for i in range(gen_len):
                y, c = mla.decode_step(x_decode[:, i : i + 1, :], c)
            assert y is not None
            return c

        mla_decode_metric, mla_decode_out = stage_benchmark(
            fn=mla_decode_full,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )
        mla_c_final = mla_decode_out
        mla_final_cache_bytes = mla_c_final.numel() * mla_c_final.element_size()

    result = {
        "seq_len": seq_len,
        "gen_len": gen_len,
        "native": {
            "prefill_time_ms": native_prefill_metric.time_s * 1000,
            "prefill_peak_mem_mb": bytes_to_mb(native_prefill_metric.peak_bytes),
            "prefill_peak_increase_mb": bytes_to_mb(native_prefill_metric.peak_increase_bytes),
            "prefill_cache_mb": bytes_to_mb(native_prefill_cache_bytes),
            "decode_time_ms": native_decode_metric.time_s * 1000,
            "decode_tokens_per_s": gen_len / max(native_decode_metric.time_s, 1e-9),
            "decode_peak_mem_mb": bytes_to_mb(native_decode_metric.peak_bytes),
            "decode_peak_increase_mb": bytes_to_mb(native_decode_metric.peak_increase_bytes),
            "decode_final_cache_mb": bytes_to_mb(native_final_cache_bytes),
        },
        "mla": {
            "prefill_time_ms": mla_prefill_metric.time_s * 1000,
            "prefill_peak_mem_mb": bytes_to_mb(mla_prefill_metric.peak_bytes),
            "prefill_peak_increase_mb": bytes_to_mb(mla_prefill_metric.peak_increase_bytes),
            "prefill_cache_mb": bytes_to_mb(mla_prefill_cache_bytes),
            "decode_time_ms": mla_decode_metric.time_s * 1000,
            "decode_tokens_per_s": gen_len / max(mla_decode_metric.time_s, 1e-9),
            "decode_peak_mem_mb": bytes_to_mb(mla_decode_metric.peak_bytes),
            "decode_peak_increase_mb": bytes_to_mb(mla_decode_metric.peak_increase_bytes),
            "decode_final_cache_mb": bytes_to_mb(mla_final_cache_bytes),
        },
    }

    native_prefill_cache = result["native"]["prefill_cache_mb"]
    mla_prefill_cache = result["mla"]["prefill_cache_mb"]
    native_decode_tps = result["native"]["decode_tokens_per_s"]
    mla_decode_tps = result["mla"]["decode_tokens_per_s"]

    result["compare"] = {
        "prefill_cache_reduction_ratio": (
            1.0 - (mla_prefill_cache / max(native_prefill_cache, 1e-9))
        ),
        "decode_cache_reduction_ratio": (
            1.0
            - (
                result["mla"]["decode_final_cache_mb"]
                / max(result["native"]["decode_final_cache_mb"], 1e-9)
            )
        ),
        "prefill_speed_ratio_native_over_mla": (
            result["native"]["prefill_time_ms"] / max(result["mla"]["prefill_time_ms"], 1e-9)
        ),
        "decode_speed_ratio_native_over_mla_tps": native_decode_tps / max(mla_decode_tps, 1e-9),
    }
    return result


def make_pairs(seq_lens: list[int], gen_lens: list[int], mode: str) -> list[tuple[int, int]]:
    if mode == "cartesian":
        return [(s, g) for s in seq_lens for g in gen_lens]
    n = min(len(seq_lens), len(gen_lens))
    return [(seq_lens[i], gen_lens[i]) for i in range(n)]


def print_summary(rows: list[dict]) -> None:
    print("\n=== Benchmark Summary ===")
    header = (
        "seq/gen | prefill(ms) native/mla | prefill_cache(MB) native/mla | "
        "decode tok/s native/mla | decode_peak(MB) native/mla"
    )
    print(header)
    for row in rows:
        seq_len = row["seq_len"]
        gen_len = row["gen_len"]
        n = row["native"]
        m = row["mla"]
        line = (
            f"{seq_len}/{gen_len} | "
            f"{n['prefill_time_ms']:.2f}/{m['prefill_time_ms']:.2f} | "
            f"{n['prefill_cache_mb']:.2f}/{m['prefill_cache_mb']:.2f} | "
            f"{n['decode_tokens_per_s']:.2f}/{m['decode_tokens_per_s']:.2f} | "
            f"{n['decode_peak_mem_mb']:.2f}/{m['decode_peak_mem_mb']:.2f}"
        )
        print(line)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print("CPU mode: force dtype float32 for compatibility.")
        dtype = torch.float32

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "N/A"

    print("=== MLA vs Native Attention Benchmark ===")
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu_name": gpu_name,
                "dtype": str(dtype).replace("torch.", ""),
                "batch_size": args.batch_size,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "latent_dim": args.latent_dim,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "pair_mode": args.pair_mode,
                "seq_lens": args.seq_lens,
                "gen_lens": args.gen_lens,
            },
            ensure_ascii=True,
        )
    )

    pairs = make_pairs(args.seq_lens, args.gen_lens, args.pair_mode)
    if not pairs:
        raise ValueError("No (seq_len, gen_len) pairs to run.")

    rows: list[dict] = []
    for idx, (seq_len, gen_len) in enumerate(pairs, start=1):
        print(f"\n[{idx}/{len(pairs)}] Running seq_len={seq_len}, gen_len={gen_len} ...")
        row = run_one_config(seq_len, gen_len, args, device, dtype)
        rows.append(row)
        print(
            "Result: "
            + json.dumps(
                {
                    "seq_len": seq_len,
                    "gen_len": gen_len,
                    "native_prefill_ms": round(row["native"]["prefill_time_ms"], 3),
                    "mla_prefill_ms": round(row["mla"]["prefill_time_ms"], 3),
                    "native_decode_tps": round(row["native"]["decode_tokens_per_s"], 3),
                    "mla_decode_tps": round(row["mla"]["decode_tokens_per_s"], 3),
                    "prefill_cache_reduction_ratio": round(
                        row["compare"]["prefill_cache_reduction_ratio"], 4
                    ),
                },
                ensure_ascii=True,
            )
        )

    output = {
        "meta": {
            "device": str(device),
            "gpu_name": gpu_name,
            "dtype": str(dtype).replace("torch.", ""),
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "latent_dim": args.latent_dim,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "pair_mode": args.pair_mode,
            "seq_lens": args.seq_lens,
            "gen_lens": args.gen_lens,
        },
        "results": rows,
    }

    print_summary(rows)
    print("\n=== JSON Result ===")
    print(json.dumps(output, indent=2, ensure_ascii=True))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n")
        print(f"\nWrote JSON to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
