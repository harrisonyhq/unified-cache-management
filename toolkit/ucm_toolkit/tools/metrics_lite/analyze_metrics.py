#!/usr/bin/env python3
"""
vLLM Metrics 聚合解析器：直接从 /metrics 拉取并统计

用法：

1. 直接从 vLLM metrics 接口拉取：
   python parse_vllm_metrics.py --url http://10.244.16.64:8000/metrics

2. 解析本地 metrics 文件：
   python parse_vllm_metrics.py --file metrics.txt

3. 同时把提取后的简化 metrics 保存下来：
   python parse_vllm_metrics.py --url http://10.244.16.64:8000/metrics --save-extracted extracted_metrics.txt
"""

import sys
import re
import argparse
import urllib.request
from pathlib import Path


# 定义需要提取的指标及其输出名称
METRICS_MAP = {
    "time_to_first_token_seconds": "TTFT (s)",
    # "inter_token_latency_seconds": "TPOT (s)",
    "e2e_request_latency_seconds": "E2E Latency (s)",
    "request_queue_time_seconds": "Queue Time (s)",
    "request_inference_time_seconds": "Inference Time (s)",
    "request_prefill_time_seconds": "Prefill Time (s)",
    "request_decode_time_seconds": "Decode Time (s)",
    "request_time_per_output_token_seconds": "Time per Output Token (s)",
}


# Cache hit rate metrics:
# base_metric_name -> (denominator_metric_name or None, display_name)
CACHE_HIT_RATE_METRICS = {
    "prefix_cache_hits": (
        "prefix_cache_queries",
        "vLLM Prefix Cache Hit Rate",
    ),
    "external_prefix_cache_hits": (
        "external_prefix_cache_queries",
        "vLLM External Prefix Cache Hit Rate",
    ),
    "interval_lookup_hit_rates": (
        None,
        "UCM Interval Lookup Hit Rate",
    ),
}


PROM_LINE_RE = re.compile(
    r"""
    ^
    (?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)
    (?:\{(?P<labels>[^}]*)\})?
    \s+
    (?P<value>
        [-+]?
        (?:
            (?:\d+(?:\.\d*)?)
            |
            (?:\.\d+)
        )
        (?:[eE][-+]?\d+)?
        |
        [-+]?Inf
        |
        NaN
    )
    """,
    re.VERBOSE,
)


def fetch_metrics(url: str, timeout: int = 10) -> str:
    """从 metrics URL 拉取 Prometheus 原始文本"""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "vllm-metrics-parser/1.0",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def read_metrics_file(file_path: str) -> str:
    """读取本地 metrics 文件"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {file_path}")
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_metric_name(raw_name: str) -> str:
    """
    归一化 metric 名称。

    例如：
      vllm:time_to_first_token_seconds_sum
        -> time_to_first_token_seconds_sum

      ucm:cache_load_shards_total
        -> cache_load_shards_total

      cache_load_shards_total
        -> cache_load_shards_total
    """
    if ":" in raw_name:
        return raw_name.split(":")[-1]
    return raw_name


def split_metric_suffix(metric_name: str):
    """
    拆分 Prometheus 指标后缀。

    输入：
      time_to_first_token_seconds_sum

    输出：
      ("time_to_first_token_seconds", "sum")

    只处理 _count / _sum / _total。
    """
    for suffix in ("_count", "_sum", "_total"):
        if metric_name.endswith(suffix):
            base_name = metric_name[: -len(suffix)]
            kind = suffix[1:]
            return base_name, kind
    return None, None


def parse_prometheus_metrics(metrics_text: str):
    """
    直接解析 Prometheus 原始 metrics 文本。

    返回：
      {
        "time_to_first_token_seconds": {
            "sum": 123.4,
            "count": 56,
        },
        "external_prefix_cache_hits": {
            "total": 523902079,
        }
      }

    注意：
    - 如果同一个指标存在多组 labels，例如不同 engine/model，
      这里会自动求和，等价于聚合所有 label。
    """
    data = {}
    extracted_lines = []

    for line in metrics_text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        match = PROM_LINE_RE.match(line)
        if not match:
            continue

        raw_name = match.group("name")
        raw_value = match.group("value")

        metric_name = normalize_metric_name(raw_name)
        base_name, kind = split_metric_suffix(metric_name)

        if not base_name or not kind:
            continue

        try:
            value = float(raw_value)
        except ValueError:
            continue

        if base_name not in data:
            data[base_name] = {}

        # 聚合不同 labels 下的同名指标
        data[base_name][kind] = data[base_name].get(kind, 0.0) + value

        # 保留一份类似 sed 输出的简化结果
        extracted_lines.append(f"{base_name}_{kind} {value}")

    return data, extracted_lines


def compute_averages(data):
    """计算各指标的平均值：sum / count"""
    results = {}

    for metric, display in METRICS_MAP.items():
        metric_data = data.get(metric)
        if not metric_data:
            results[display] = None
            continue

        count = metric_data.get("count")
        total = metric_data.get("sum")

        if count and count > 0 and total is not None:
            results[display] = total / count
        else:
            results[display] = None

    return results


def compute_throughput(data):
    """
    计算 TPS 相关指标。

    注意：
    这里的 Decode TPS = 总生成 token 数 / request_decode_time_seconds_sum

    它不是严格意义上的 wall-clock TPS。
    如果你想计算某个采样周期内真实吞吐，需要对 counter 做两次采样后取 diff。
    """
    tps = {}

    gen_tokens = data.get("request_generation_tokens", {}).get("sum")
    prefill_time = data.get("request_prefill_time_seconds", {}).get("sum")
    decode_time = data.get("request_decode_time_seconds", {}).get("sum")

    if gen_tokens and gen_tokens > 0:
        if decode_time and decode_time > 0:
            tps["Decode TPS"] = gen_tokens / decode_time

        if (
            prefill_time is not None
            and decode_time is not None
            and (prefill_time + decode_time) > 0
        ):
            tps["Overall TPS"] = gen_tokens / (prefill_time + decode_time)

    return tps


def compute_cache_hit_rates(data):
    """计算缓存命中率指标"""
    hit_rates = {}

    for base_metric, (denom_or_none, display_name) in CACHE_HIT_RATE_METRICS.items():
        metric_data = data.get(base_metric)

        if not metric_data:
            hit_rates[display_name] = None
            continue

        # UCM interval_lookup_hit_rates: sum / count
        if denom_or_none is None:
            sum_val = metric_data.get("sum")
            count_val = metric_data.get("count")

            if count_val and count_val > 0 and sum_val is not None:
                hit_rates[display_name] = sum_val / count_val
            else:
                hit_rates[display_name] = None

            continue

        # counter 类型 hit rate: hits_total / queries_total
        denom_data = data.get(denom_or_none)
        if not denom_data:
            hit_rates[display_name] = None
            continue

        num_val = metric_data.get("total", 0.0)
        denom_val = denom_data.get("total", 0.0)

        if denom_val and denom_val > 0:
            hit_rates[display_name] = num_val / denom_val
        else:
            hit_rates[display_name] = None

    # UCM cache load backend ratio:
    # cache_load_backend_shards_total / cache_load_shards_total
    backend_val = data.get("cache_load_backend_shards", {}).get("total")
    shards_val = data.get("cache_load_shards", {}).get("total")

    if backend_val is not None and shards_val and shards_val > 0:
        backend_ratio = backend_val / shards_val
        hit_rates["UCM Cache Load Backend Ratio (POSIX Load)"] = backend_ratio
        hit_rates["UCM Cache Hit Rate (in cache)"] = 1 - backend_ratio

    return hit_rates


def print_summary(averages, tps, total_requests, total_tokens, cache_hit_rates=None):
    """格式化输出摘要"""
    print("\n=== vLLM Metrics Summary ===")
    print(f"Total requests: {total_requests}")
    print(f"Total generated tokens: {total_tokens}")

    print("\nAverage Latencies:")
    for display, value in averages.items():
        if value is not None:
            print(f"  {display}: {value:.4f}")
        else:
            print(f"  {display}: N/A")

    if tps:
        print("\nThroughput (TPS):")
        for name, value in tps.items():
            print(f"  {name}: {value:.2f} tokens/s")

    if cache_hit_rates:
        print("\nCache Hit Rates:")
        for name, value in cache_hit_rates.items():
            if value is not None:
                print(f"  {name}: {value:.4f} ({value * 100:.2f}%)")
            else:
                print(f"  {name}: N/A")

    print("=" * 32)


def main():
    parser = argparse.ArgumentParser(
        description="Parse vLLM Prometheus metrics and print summary."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--url",
        help="vLLM metrics URL, e.g. http://10.244.16.64:8000/metrics",
    )
    source.add_argument(
        "--file",
        help="local metrics file path",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP timeout seconds when using --url. Default: 10",
    )

    parser.add_argument(
        "--save-extracted",
        help="save simplified extracted metrics, similar to curl + sed output",
    )

    args = parser.parse_args()

    try:
        if args.url:
            metrics_text = fetch_metrics(args.url, timeout=args.timeout)
        else:
            metrics_text = read_metrics_file(args.file)
    except Exception as e:
        print(f"Error: failed to load metrics: {e}", file=sys.stderr)
        sys.exit(1)

    data, extracted_lines = parse_prometheus_metrics(metrics_text)

    if args.save_extracted:
        Path(args.save_extracted).write_text(
            "\n".join(extracted_lines) + "\n",
            encoding="utf-8",
        )

    total_reqs = int(data.get("time_to_first_token_seconds", {}).get("count", 0))
    total_gen_tokens = int(data.get("request_generation_tokens", {}).get("sum", 0))

    averages = compute_averages(data)
    tps = compute_throughput(data)
    cache_hit_rates = compute_cache_hit_rates(data)

    print_summary(
        averages=averages,
        tps=tps,
        total_requests=total_reqs,
        total_tokens=total_gen_tokens,
        cache_hit_rates=cache_hit_rates,
    )


if __name__ == "__main__":
    main()