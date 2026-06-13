#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight Grafana-like metrics collector for vLLM/UCM Prometheus endpoints.

The collector intentionally keeps the hot path small:
1. scrape the /metrics endpoint at a fixed interval;
2. parse only the raw samples required by the configured formulas;
3. write those selected raw samples to a gzip CSV for auditability;
4. compute interval values and whole-run summary values with PromQL-like delta
   formulas.

Outputs:
  - timeseries.csv: interval-level metrics.
  - summary.csv: whole-run weighted averages/rates/ratios.
  - raw_selected_samples.csv.gz: only raw samples used by formulas.
  - scrape_meta.csv: request timing, size, and parse status.
  - manifest.json: formulas and collection options.
  - plots/: optional PNG charts generated after collection.

Example:
  python metrics_lite.py \
    --url http://127.0.0.1:8000/metrics \
    --interval 5 \
    --duration 600 \
    --out ./vllm_metrics_result \
    --model-name Qwen \
    --continue-on-error
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import signal
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


VLLM_METRIC_PREFIX = "vllm:"


HISTOGRAM_AVG_SPECS: List[Tuple[str, str, str]] = [
    (
        "e2e_request_latency_avg_s",
        "vllm:e2e_request_latency_seconds",
        "delta(sum) / delta(count), equivalent to rate(_sum) / rate(_count)",
    ),
    (
        "ttft_avg_s",
        "vllm:time_to_first_token_seconds",
        "delta(sum) / delta(count), equivalent to rate(_sum) / rate(_count)",
    ),
    (
        "tpot_avg_s",
        "vllm:request_time_per_output_token_seconds",
        "delta(sum) / delta(count), equivalent to rate(_sum) / rate(_count)",
    ),
]


COUNTER_RATE_SPECS: List[Tuple[str, str, str, str]] = [
    (
        "prompt_tokens_per_s",
        "vllm:prompt_tokens_total",
        "tokens/s",
        "delta(counter) / delta(time), equivalent to rate(counter)",
    ),
    (
        "generation_tokens_per_s",
        "vllm:generation_tokens_total",
        "tokens/s",
        "delta(counter) / delta(time), equivalent to rate(counter)",
    ),
]


RATIO_SPECS: List[Tuple[str, str, str, Optional[float], str]] = [
    (
        "prefix_cache_hit_rate",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        None,
        "delta(hits) / delta(queries), equivalent to rate(hits) / rate(queries)",
    ),
    (
        "external_prefix_cache_hit_rate",
        "vllm:external_prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total",
        None,
        "delta(hits) / delta(queries), equivalent to rate(hits) / rate(queries)",
    ),
    (
        "posix_store_load_ratio",
        "ucm:cache_load_backend_shards_total",
        "ucm:cache_load_shards_total",
        1.0,
        "rate(cache_load_backend_shards_total) / clamp_min(rate(cache_load_shards_total), 1); shards that missed Cache Store and loaded from Posix/backend",
    ),
    (
        "posix_store_load_fraction",
        "ucm:cache_load_backend_shards_total",
        "ucm:cache_load_shards_total",
        None,
        "delta(cache_load_backend_shards_total) / delta(cache_load_shards_total); shards that missed Cache Store and loaded from Posix/backend",
    ),
]


DERIVED_COLUMNS = (
    [name for name, _, _ in HISTOGRAM_AVG_SPECS]
    + [name for name, _, _, _ in COUNTER_RATE_SPECS]
    + [name for name, _, _, _, _ in RATIO_SPECS]
    + ["cache_store_load_hit_fraction"]
)


PLOT_GROUPS: List[Tuple[str, List[str], str, str]] = [
    (
        "e2e_request_latency_avg",
        ["e2e_request_latency_avg_s"],
        "E2E Request Latency Avg",
        "seconds",
    ),
    (
        "ttft_avg",
        ["ttft_avg_s"],
        "TTFT Avg",
        "seconds",
    ),
    (
        "tpot_avg",
        ["tpot_avg_s"],
        "TPOT Avg",
        "seconds",
    ),
    (
        "prompt_token_throughput",
        ["prompt_tokens_per_s"],
        "Prompt Token Throughput",
        "tokens/s",
    ),
    (
        "generation_token_throughput",
        ["generation_tokens_per_s"],
        "Generation Token Throughput",
        "tokens/s",
    ),
    (
        "cache_hit_rates",
        ["prefix_cache_hit_rate", "external_prefix_cache_hit_rate"],
        "Prefix Cache Hit Rates",
        "ratio",
    ),
    (
        "cache_store_vs_posix_store_load",
        ["posix_store_load_ratio", "posix_store_load_fraction", "cache_store_load_hit_fraction"],
        "Cache Store vs Posix Store Load Source",
        "ratio",
    ),
]


STOP_REQUESTED = False


SAMPLE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(\{([^}]*)\})?'
    r'\s+'
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)'
    r'(?:\s+\d+)?'
    r'\s*$'
)

LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


@dataclass
class MetricSample:
    metric: str
    labels: Dict[str, str]
    value: float


@dataclass
class ScrapeResult:
    text: str
    timestamp: float
    ok: bool
    status_code: str
    duration_seconds: float
    response_bytes: int
    error: str = ""


@dataclass
class LabelFilters:
    exact: Dict[str, str]
    regex: Dict[str, re.Pattern[str]]
    contains: Optional[str] = None

    def matches(self, labels: Dict[str, str], labels_json: str) -> bool:
        if self.contains and self.contains not in labels_json:
            return False
        for key, expected in self.exact.items():
            if labels.get(key) != expected:
                return False
        for key, pattern in self.regex.items():
            if not pattern.search(labels.get(key, "")):
                return False
        return True


class SummaryAccumulator:
    def __init__(self) -> None:
        self.hist_sum: Dict[str, float] = defaultdict(float)
        self.hist_count: Dict[str, float] = defaultdict(float)
        self.counter_delta: Dict[str, float] = defaultdict(float)
        self.counter_seconds: Dict[str, float] = defaultdict(float)
        self.ratio_num: Dict[str, float] = defaultdict(float)
        self.ratio_den: Dict[str, float] = defaultdict(float)
        self.ratio_seconds: Dict[str, float] = defaultdict(float)
        self.valid_intervals: Dict[str, int] = defaultdict(int)

    def add_hist(self, name: str, delta_sum: float, delta_count: float) -> None:
        self.hist_sum[name] += delta_sum
        self.hist_count[name] += delta_count
        self.valid_intervals[name] += 1

    def add_counter(self, name: str, delta_value: float, seconds: float) -> None:
        self.counter_delta[name] += delta_value
        self.counter_seconds[name] += seconds
        self.valid_intervals[name] += 1

    def add_ratio(self, name: str, delta_num: float, delta_den: float, seconds: float) -> None:
        self.ratio_num[name] += delta_num
        self.ratio_den[name] += delta_den
        self.ratio_seconds[name] += seconds
        self.valid_intervals[name] += 1


def request_stop(signum=None, frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    if signum is not None:
        print(f"\n[INFO] received signal {signum}; will stop after current scrape")


def normalize_metric_name(name: str, default_prefix: str = VLLM_METRIC_PREFIX) -> str:
    if ":" in name:
        return name
    return f"{default_prefix}{name}"


def counter_candidate_names(metric_name: str) -> List[str]:
    metric = normalize_metric_name(metric_name)
    candidates = [metric]
    if metric.endswith("_total"):
        candidates.append(metric[:-6])
    else:
        candidates.append(f"{metric}_total")

    seen = set()
    result = []
    for candidate in candidates:
        if candidate not in seen:
            result.append(candidate)
            seen.add(candidate)
    return result


def required_metric_names() -> List[str]:
    names: List[str] = []
    for _, base_metric, _ in HISTOGRAM_AVG_SPECS:
        names.append(f"{base_metric}_sum")
        names.append(f"{base_metric}_count")
    for _, metric, _, _ in COUNTER_RATE_SPECS:
        names.extend(counter_candidate_names(metric))
    for _, numerator, denominator, _, _ in RATIO_SPECS:
        names.extend(counter_candidate_names(numerator))
        names.extend(counter_candidate_names(denominator))

    seen = set()
    result = []
    for name in names:
        if name not in seen:
            result.append(name)
            seen.add(name)
    return result


def parse_labels(label_str: Optional[str]) -> Dict[str, str]:
    if not label_str:
        return {}

    labels: Dict[str, str] = {}
    for key, value in LABEL_RE.findall(label_str):
        labels[key] = (
            value.replace(r"\\", "\\")
            .replace(r"\"", '"')
            .replace(r"\n", "\n")
        )
    return labels


def labels_to_json(labels: Dict[str, str]) -> str:
    return json.dumps(labels, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_selected_samples(
    text: str,
    wanted_names: Sequence[str],
    filters: LabelFilters,
) -> List[MetricSample]:
    wanted = set(wanted_names)
    samples: List[MetricSample] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        match = SAMPLE_RE.match(line)
        if not match:
            continue

        metric = match.group(1)
        if metric not in wanted:
            continue

        try:
            value = float(match.group(4))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue

        labels = parse_labels(match.group(3))
        labels_json = labels_to_json(labels)
        if not filters.matches(labels, labels_json):
            continue

        samples.append(MetricSample(metric=metric, labels=labels, value=value))

    return samples


def aggregate_samples(samples: Iterable[MetricSample]) -> Dict[str, float]:
    values: Dict[str, float] = defaultdict(float)
    for sample in samples:
        values[sample.metric] += sample.value
    return dict(values)


def pick_value(values: Dict[str, float], metric_name: str) -> Optional[float]:
    for candidate in counter_candidate_names(metric_name):
        if candidate in values:
            return values[candidate]
    return None


def delta_value(
    previous: Dict[str, float],
    current: Dict[str, float],
    metric_name: str,
) -> Optional[float]:
    prev = pick_value(previous, metric_name)
    cur = pick_value(current, metric_name)
    if prev is None or cur is None:
        return None
    delta = cur - prev
    if delta < 0:
        return None
    return delta


def safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def ratio_from_deltas(
    delta_num: float,
    delta_den: float,
    seconds: float,
    denominator_rate_clamp_min: Optional[float],
) -> Optional[float]:
    if seconds <= 0:
        return None
    if denominator_rate_clamp_min is None:
        return safe_div(delta_num, delta_den)

    numerator_rate = delta_num / seconds
    denominator_rate = delta_den / seconds
    denominator_rate = max(denominator_rate, denominator_rate_clamp_min)
    return safe_div(numerator_rate, denominator_rate)


def compute_interval_row(
    scrape_index: int,
    timestamp: float,
    elapsed_seconds: float,
    interval_seconds: float,
    previous: Dict[str, float],
    current: Dict[str, float],
    summary: SummaryAccumulator,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "scrape_index": scrape_index,
        "timestamp": f"{timestamp:.6f}",
        "datetime": datetime.fromtimestamp(timestamp).isoformat(timespec="seconds"),
        "elapsed_seconds": f"{elapsed_seconds:.6f}",
        "interval_seconds": f"{interval_seconds:.6f}",
    }

    for output_name, base_metric, _ in HISTOGRAM_AVG_SPECS:
        delta_sum = delta_value(previous, current, f"{base_metric}_sum")
        delta_count = delta_value(previous, current, f"{base_metric}_count")
        value = safe_div(delta_sum, delta_count)
        if value is not None and delta_sum is not None and delta_count is not None:
            summary.add_hist(output_name, delta_sum, delta_count)
        row[output_name] = format_optional(value)

    for output_name, metric_name, _, _ in COUNTER_RATE_SPECS:
        delta = delta_value(previous, current, metric_name)
        value = safe_div(delta, interval_seconds)
        if value is not None and delta is not None:
            summary.add_counter(output_name, delta, interval_seconds)
        row[output_name] = format_optional(value)

    posix_fraction: Optional[float] = None
    for output_name, numerator, denominator, clamp_min, _ in RATIO_SPECS:
        delta_num = delta_value(previous, current, numerator)
        delta_den = delta_value(previous, current, denominator)
        value = None
        if delta_num is not None and delta_den is not None:
            value = ratio_from_deltas(delta_num, delta_den, interval_seconds, clamp_min)
            if value is not None:
                summary.add_ratio(output_name, delta_num, delta_den, interval_seconds)
        if output_name == "posix_store_load_fraction":
            posix_fraction = value
        row[output_name] = format_optional(value)

    store_fraction = None
    if posix_fraction is not None:
        store_fraction = max(0.0, min(1.0, 1.0 - posix_fraction))
        summary.add_ratio(
            "cache_store_load_hit_fraction",
            max(0.0, (delta_value(previous, current, "ucm:cache_load_shards_total") or 0.0)
                - (delta_value(previous, current, "ucm:cache_load_backend_shards_total") or 0.0)),
            delta_value(previous, current, "ucm:cache_load_shards_total") or 0.0,
            interval_seconds,
        )
    row["cache_store_load_hit_fraction"] = format_optional(store_fraction)

    return row


def format_optional(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def write_manifest(out_dir: Path, args: argparse.Namespace, wanted_names: Sequence[str]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": {
            "timeseries": "timeseries.csv",
            "summary": "summary.csv",
            "raw_selected_samples": "raw_selected_samples.csv.gz",
            "scrape_meta": "scrape_meta.csv",
        },
        "selected_raw_metrics": list(wanted_names),
        "histogram_average_specs": [
            {"name": name, "base_metric": base, "method": method}
            for name, base, method in HISTOGRAM_AVG_SPECS
        ],
        "counter_rate_specs": [
            {"name": name, "metric": metric, "unit": unit, "method": method}
            for name, metric, unit, method in COUNTER_RATE_SPECS
        ],
        "ratio_specs": [
            {
                "name": name,
                "numerator": numerator,
                "denominator": denominator,
                "denominator_rate_clamp_min": clamp_min,
                "method": method,
            }
            for name, numerator, denominator, clamp_min, method in RATIO_SPECS
        ],
        "derived_metrics": {
            "cache_store_load_hit_fraction": "1 - posix_store_load_fraction, clamped to [0, 1]; shards served by Cache Store without loading from Posix/backend",
        },
        "options": {
            "url": getattr(args, "url", None),
            "interval": args.interval,
            "duration": args.duration,
            "timeout": args.timeout,
            "labels_contains": args.labels_contains,
            "model_name": args.model_name,
            "job_regex": args.job_regex,
            "worker_id_regex": args.worker_id_regex,
            "instance_regex": args.instance_regex,
            "label": args.label,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_summary(out_dir: Path, summary: SummaryAccumulator) -> None:
    path = out_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "value",
                "unit",
                "method",
                "valid_intervals",
                "numerator_delta",
                "denominator_delta",
                "total_seconds",
            ],
        )
        writer.writeheader()

        for name, base_metric, method in HISTOGRAM_AVG_SPECS:
            numerator = summary.hist_sum.get(name, 0.0)
            denominator = summary.hist_count.get(name, 0.0)
            writer.writerow(
                {
                    "metric": name,
                    "value": format_optional(safe_div(numerator, denominator)),
                    "unit": "seconds",
                    "method": method,
                    "valid_intervals": summary.valid_intervals.get(name, 0),
                    "numerator_delta": format_optional(numerator),
                    "denominator_delta": format_optional(denominator),
                    "total_seconds": "",
                }
            )

        for name, _, unit, method in COUNTER_RATE_SPECS:
            numerator = summary.counter_delta.get(name, 0.0)
            seconds = summary.counter_seconds.get(name, 0.0)
            writer.writerow(
                {
                    "metric": name,
                    "value": format_optional(safe_div(numerator, seconds)),
                    "unit": unit,
                    "method": method,
                    "valid_intervals": summary.valid_intervals.get(name, 0),
                    "numerator_delta": format_optional(numerator),
                    "denominator_delta": "",
                    "total_seconds": format_optional(seconds),
                }
            )

        for name, _, _, clamp_min, method in RATIO_SPECS:
            numerator = summary.ratio_num.get(name, 0.0)
            denominator = summary.ratio_den.get(name, 0.0)
            seconds = summary.ratio_seconds.get(name, 0.0)
            value = ratio_from_deltas(numerator, denominator, seconds, clamp_min)
            writer.writerow(
                {
                    "metric": name,
                    "value": format_optional(value),
                    "unit": "ratio",
                    "method": method,
                    "valid_intervals": summary.valid_intervals.get(name, 0),
                    "numerator_delta": format_optional(numerator),
                    "denominator_delta": format_optional(denominator),
                    "total_seconds": format_optional(seconds),
                }
            )

        name = "cache_store_load_hit_fraction"
        numerator = summary.ratio_num.get(name, 0.0)
        denominator = summary.ratio_den.get(name, 0.0)
        writer.writerow(
            {
                "metric": name,
                "value": format_optional(safe_div(numerator, denominator)),
                "unit": "ratio",
                "method": "delta(cache_load_shards_total - cache_load_backend_shards_total) / delta(cache_load_shards_total); shards served by Cache Store without loading from Posix/backend",
                "valid_intervals": summary.valid_intervals.get(name, 0),
                "numerator_delta": format_optional(numerator),
                "denominator_delta": format_optional(denominator),
                "total_seconds": format_optional(summary.ratio_seconds.get(name, 0.0)),
            }
        )


def parse_float_cell(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def read_timeseries(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row: Dict[str, object] = {}
            for key, value in raw_row.items():
                if key in ("datetime",):
                    row[key] = value
                    continue
                row[key] = parse_float_cell(value)
            rows.append(row)
    return rows


def row_float(row: Dict[str, object], key: str) -> Optional[float]:
    value = row.get(key)
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    return None


def format_time_label(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def plot_x_values(rows: List[Dict[str, object]]) -> List[float]:
    first_timestamp = row_float(rows[0], "timestamp") if rows else None
    values: List[float] = []
    for row in rows:
        elapsed = row_float(row, "elapsed_seconds")
        if elapsed is not None:
            values.append(elapsed)
            continue

        timestamp = row_float(row, "timestamp")
        if timestamp is not None and first_timestamp is not None:
            values.append(timestamp - first_timestamp)
        else:
            values.append(float(len(values)))
    return values


def plot_tick_values(
    rows: List[Dict[str, object]],
    x_values: List[float],
    max_ticks: int = 6,
) -> List[Tuple[float, str]]:
    if not rows or not x_values:
        return []

    if len(rows) <= max_ticks:
        indices = list(range(len(rows)))
    else:
        indices = sorted({
            round(i * (len(rows) - 1) / (max_ticks - 1))
            for i in range(max_ticks)
        })

    ticks: List[Tuple[float, str]] = []
    for idx in indices:
        timestamp = row_float(rows[idx], "timestamp")
        if timestamp is None:
            label = format_optional(x_values[idx])
        else:
            label = format_time_label(timestamp)
        ticks.append((x_values[idx], label))
    return ticks


def write_svg_plot(
    out_path: Path,
    rows: List[Dict[str, object]],
    columns: List[str],
    title: str,
    ylabel: str,
) -> bool:
    from xml.sax.saxutils import escape

    x_values = plot_x_values(rows)
    tick_values = plot_tick_values(rows, x_values)
    series: List[Tuple[str, List[Tuple[float, float]]]] = []
    y_values: List[float] = []

    for column in columns:
        points = [
            (x, row_float(row, column))
            for x, row in zip(x_values, rows)
            if row_float(row, column) is not None
        ]
        clean_points = [(x, y) for x, y in points if y is not None]
        if clean_points:
            series.append((column, clean_points))
            y_values.extend(y for _, y in clean_points)

    if not series or not y_values:
        return False

    width = 1000
    height = 520
    left = 86
    right = 24
    top = 54
    bottom = 112
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    y_min = min(0.0, min(y_values))
    y_max = max(y_values)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    elif y_max <= 1.0 and y_min >= 0.0:
        y_max = 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_height

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" fill="#111827">{escape(title)}</text>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13" fill="#374151">Sample time (HH:MM:SS)</text>',
        f'<text x="18" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="13" fill="#374151" transform="rotate(-90 18 {height / 2})">{escape(ylabel)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151" stroke-width="1"/>',
    ]

    for i in range(6):
        y = top + i * plot_height / 5
        value = y_max - i * (y_max - y_min) / 5
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{value:.4g}</text>')

    for value, label in tick_values:
        x = sx(value)
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 18}" text-anchor="end" font-family="Arial" font-size="10" fill="#4b5563" transform="rotate(-30 {x:.2f} {top + plot_height + 18})">{escape(label)}</text>')

    legend_x = left
    legend_y = height - 34
    for idx, (name, points) in enumerate(series):
        color = colors[idx % len(colors)]
        polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>')
        lx = legend_x + (idx % 2) * 420
        ly = legend_y + (idx // 2) * 18
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 18}" y="{ly}" font-family="Arial" font-size="12" fill="#111827">{escape(name)}</text>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return True


def generate_plots(out_dir: Path) -> None:
    timeseries_path = out_dir / "timeseries.csv"
    if not timeseries_path.exists():
        raise FileNotFoundError(f"timeseries file not found: {timeseries_path}")

    rows = read_timeseries(timeseries_path)
    if not rows:
        print(f"[WARN] no interval rows in {timeseries_path}; no plots generated")
        return

    plt = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib is not available; generating SVG plots instead: {exc}")

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    x_values = plot_x_values(rows)
    tick_values = plot_tick_values(rows, x_values)

    generated = 0
    for filename, columns, title, ylabel in PLOT_GROUPS:
        if plt is None:
            if write_svg_plot(plot_dir / f"{filename}.svg", rows, columns, title, ylabel):
                generated += 1
            else:
                print(f"[WARN] no valid data for plot: {title}")
            continue

        has_series = False
        plt.figure(figsize=(12, 6))
        for column in columns:
            points = [
                (x, row_float(row, column))
                for x, row in zip(x_values, rows)
                if row_float(row, column) is not None
            ]
            if not points:
                continue

            xs, ys = zip(*points)
            plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=column)
            has_series = True

        if not has_series:
            plt.close()
            print(f"[WARN] no valid data for plot: {title}")
            continue

        ax = plt.gca()
        if tick_values:
            ax.set_xticks([x for x, _ in tick_values])
            ax.set_xticklabels([label for _, label in tick_values])
        plt.xlabel("Sample time (HH:MM:SS)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        out_path = plot_dir / f"{filename}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        generated += 1

    print(f"[INFO] generated {generated} plot(s) in: {plot_dir}")


def scrape_http_once(url: str, timeout: int) -> ScrapeResult:
    started = time.time()
    try:
        import requests

        response = requests.get(url, timeout=timeout)
        duration = time.time() - started
        text = response.text
        response.raise_for_status()
        return ScrapeResult(
            text=text,
            timestamp=time.time(),
            ok=True,
            status_code=str(response.status_code),
            duration_seconds=duration,
            response_bytes=len(response.content),
        )
    except Exception as exc:
        duration = time.time() - started
        return ScrapeResult(
            text="",
            timestamp=time.time(),
            ok=False,
            status_code="",
            duration_seconds=duration,
            response_bytes=0,
            error=str(exc),
        )


def http_scrape_loop(args: argparse.Namespace) -> Iterator[ScrapeResult]:
    global STOP_REQUESTED
    STOP_REQUESTED = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    start = time.time()
    stop_file = Path(args.stop_file) if args.stop_file else None

    while True:
        now = time.time()
        elapsed = now - start

        if STOP_REQUESTED:
            print("[INFO] stop requested; exiting scrape loop")
            break
        if stop_file and stop_file.exists():
            print(f"[INFO] stop-file detected: {stop_file}")
            break
        if args.duration > 0 and elapsed > args.duration:
            print("[INFO] duration reached; exiting scrape loop")
            break

        yield scrape_http_once(args.url, args.timeout)

        if args.once:
            break

        slept = 0.0
        sleep_step = min(0.5, args.interval)
        while slept < args.interval:
            if STOP_REQUESTED:
                break
            if stop_file and stop_file.exists():
                STOP_REQUESTED = True
                break
            remaining = args.interval - slept
            step = min(sleep_step, remaining)
            time.sleep(step)
            slept += step


def replay_scrape_loop(args: argparse.Namespace) -> Iterator[ScrapeResult]:
    start = time.time()
    for idx, file_name in enumerate(args.replay_files):
        path = Path(file_name)
        started = time.time()
        text = path.read_text(encoding="utf-8")
        yield ScrapeResult(
            text=text,
            timestamp=start + idx * args.interval,
            ok=True,
            status_code="replay",
            duration_seconds=time.time() - started,
            response_bytes=len(text.encode("utf-8")),
        )


def collect_from_scrapes(
    scrapes: Iterable[ScrapeResult],
    args: argparse.Namespace,
    filters: LabelFilters,
) -> SummaryAccumulator:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_names = required_metric_names()
    write_manifest(out_dir, args, wanted_names)

    summary = SummaryAccumulator()
    previous_values: Optional[Dict[str, float]] = None
    previous_timestamp: Optional[float] = None
    start_timestamp: Optional[float] = None

    raw_path = out_dir / "raw_selected_samples.csv.gz"
    meta_path = out_dir / "scrape_meta.csv"
    timeseries_path = out_dir / "timeseries.csv"

    with gzip.open(raw_path, "wt", encoding="utf-8", newline="") as raw_file, \
        meta_path.open("w", encoding="utf-8", newline="") as meta_file, \
        timeseries_path.open("w", encoding="utf-8", newline="") as timeseries_file:

        raw_writer = csv.DictWriter(
            raw_file,
            fieldnames=["scrape_index", "timestamp", "datetime", "metric", "labels_json", "value"],
        )
        meta_writer = csv.DictWriter(
            meta_file,
            fieldnames=[
                "scrape_index",
                "timestamp",
                "datetime",
                "elapsed_seconds",
                "ok",
                "status_code",
                "scrape_duration_seconds",
                "response_bytes",
                "selected_samples",
                "sha256",
                "error",
            ],
        )
        timeseries_writer = csv.DictWriter(
            timeseries_file,
            fieldnames=[
                "scrape_index",
                "timestamp",
                "datetime",
                "elapsed_seconds",
                "interval_seconds",
                *DERIVED_COLUMNS,
            ],
        )
        raw_writer.writeheader()
        meta_writer.writeheader()
        timeseries_writer.writeheader()

        for scrape_index, scrape in enumerate(scrapes):
            if start_timestamp is None:
                start_timestamp = scrape.timestamp
            elapsed_seconds = scrape.timestamp - start_timestamp

            selected_samples: List[MetricSample] = []
            current_values: Dict[str, float] = {}
            sha256 = ""

            if scrape.ok:
                sha256 = hashlib.sha256(scrape.text.encode("utf-8")).hexdigest()
                selected_samples = parse_selected_samples(scrape.text, wanted_names, filters)
                current_values = aggregate_samples(selected_samples)

                for sample in selected_samples:
                    raw_writer.writerow(
                        {
                            "scrape_index": scrape_index,
                            "timestamp": f"{scrape.timestamp:.6f}",
                            "datetime": datetime.fromtimestamp(scrape.timestamp).isoformat(timespec="seconds"),
                            "metric": sample.metric,
                            "labels_json": labels_to_json(sample.labels),
                            "value": format_optional(sample.value),
                        }
                    )

            meta_writer.writerow(
                {
                    "scrape_index": scrape_index,
                    "timestamp": f"{scrape.timestamp:.6f}",
                    "datetime": datetime.fromtimestamp(scrape.timestamp).isoformat(timespec="seconds"),
                    "elapsed_seconds": f"{elapsed_seconds:.6f}",
                    "ok": scrape.ok,
                    "status_code": scrape.status_code,
                    "scrape_duration_seconds": f"{scrape.duration_seconds:.6f}",
                    "response_bytes": scrape.response_bytes,
                    "selected_samples": len(selected_samples),
                    "sha256": sha256,
                    "error": scrape.error,
                }
            )

            if not scrape.ok:
                print(f"[WARN] scrape #{scrape_index} failed: {scrape.error}")
                if not args.continue_on_error:
                    raise RuntimeError(scrape.error)
                continue

            if previous_values is not None and previous_timestamp is not None:
                interval_seconds = scrape.timestamp - previous_timestamp
                row = compute_interval_row(
                    scrape_index=scrape_index,
                    timestamp=scrape.timestamp,
                    elapsed_seconds=elapsed_seconds,
                    interval_seconds=interval_seconds,
                    previous=previous_values,
                    current=current_values,
                    summary=summary,
                )
                timeseries_writer.writerow(row)

            previous_values = current_values
            previous_timestamp = scrape.timestamp

            print(
                f"[INFO] scrape #{scrape_index}: selected_samples={len(selected_samples)}, "
                f"elapsed={elapsed_seconds:.1f}s"
            )

    write_summary(out_dir, summary)

    if args.plot:
        generate_plots(out_dir)

    print(f"[INFO] timeseries saved to: {timeseries_path}")
    print(f"[INFO] summary saved to: {out_dir / 'summary.csv'}")
    print(f"[INFO] selected raw samples saved to: {raw_path}")
    print(f"[INFO] scrape metadata saved to: {meta_path}")

    return summary


def build_label_filters(args: argparse.Namespace) -> LabelFilters:
    exact: Dict[str, str] = {}
    regex: Dict[str, re.Pattern[str]] = {}

    if args.model_name:
        exact["model_name"] = args.model_name
    if args.job_regex:
        regex["job"] = re.compile(args.job_regex)
    if args.worker_id_regex:
        regex["worker_id"] = re.compile(args.worker_id_regex)
    if args.instance_regex:
        regex["instance"] = re.compile(args.instance_regex)

    for item in args.label or []:
        if "=" not in item:
            raise ValueError(f"--label expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        exact[key] = value

    return LabelFilters(exact=exact, regex=regex, contains=args.labels_contains)


def run_collection(args: argparse.Namespace) -> SummaryAccumulator:
    if args.replay_files:
        scrapes = replay_scrape_loop(args)
    else:
        if not args.url:
            raise ValueError("--url is required unless --replay-files or --self-test is used")
        scrapes = http_scrape_loop(args)

    filters = build_label_filters(args)
    return collect_from_scrapes(scrapes, args, filters)


def self_test_texts() -> List[str]:
    first = """
# HELP vllm:e2e_request_latency_seconds request latency
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_sum{model_name="m",worker_id="0"} 10
vllm:e2e_request_latency_seconds_count{model_name="m",worker_id="0"} 5
vllm:e2e_request_latency_seconds_sum{model_name="other",worker_id="0"} 1000
vllm:e2e_request_latency_seconds_count{model_name="other",worker_id="0"} 100
vllm:time_to_first_token_seconds_sum{model_name="m",worker_id="0"} 2
vllm:time_to_first_token_seconds_count{model_name="m",worker_id="0"} 5
vllm:request_time_per_output_token_seconds_sum{model_name="m",worker_id="0"} 1
vllm:request_time_per_output_token_seconds_count{model_name="m",worker_id="0"} 5
vllm:prompt_tokens_total{model_name="m",worker_id="0"} 100
vllm:generation_tokens_total{model_name="m",worker_id="0"} 200
vllm:prefix_cache_hits_total{model_name="m",worker_id="0"} 40
vllm:prefix_cache_queries_total{model_name="m",worker_id="0"} 50
vllm:external_prefix_cache_hits_total{model_name="m",worker_id="0"} 10
vllm:external_prefix_cache_queries_total{model_name="m",worker_id="0"} 25
ucm:cache_load_backend_shards_total{model_name="m",worker_id="0"} 6
ucm:cache_load_shards_total{model_name="m",worker_id="0"} 10
"""
    second = """
vllm:e2e_request_latency_seconds_sum{model_name="m",worker_id="0"} 18
vllm:e2e_request_latency_seconds_count{model_name="m",worker_id="0"} 9
vllm:e2e_request_latency_seconds_sum{model_name="other",worker_id="0"} 2000
vllm:e2e_request_latency_seconds_count{model_name="other",worker_id="0"} 200
vllm:time_to_first_token_seconds_sum{model_name="m",worker_id="0"} 3.2
vllm:time_to_first_token_seconds_count{model_name="m",worker_id="0"} 9
vllm:request_time_per_output_token_seconds_sum{model_name="m",worker_id="0"} 1.8
vllm:request_time_per_output_token_seconds_count{model_name="m",worker_id="0"} 9
vllm:prompt_tokens_total{model_name="m",worker_id="0"} 160
vllm:generation_tokens_total{model_name="m",worker_id="0"} 300
vllm:prefix_cache_hits_total{model_name="m",worker_id="0"} 70
vllm:prefix_cache_queries_total{model_name="m",worker_id="0"} 100
vllm:external_prefix_cache_hits_total{model_name="m",worker_id="0"} 20
vllm:external_prefix_cache_queries_total{model_name="m",worker_id="0"} 50
ucm:cache_load_backend_shards_total{model_name="m",worker_id="0"} 16
ucm:cache_load_shards_total{model_name="m",worker_id="0"} 30
"""
    return [first, second]


def assert_close(actual: float, expected: float, name: str) -> None:
    if abs(actual - expected) > 1e-9:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="metrics_lite_selftest_") as tmp:
        args = argparse.Namespace(
            out=tmp,
            interval=10.0,
            duration=0.0,
            timeout=10,
            labels_contains=None,
            model_name="m",
            job_regex=None,
            worker_id_regex=None,
            instance_regex=None,
            label=[],
            url=None,
            replay_files=None,
            stop_file=None,
            once=False,
            continue_on_error=False,
            plot=True,
            plot_only=None,
        )

        start = time.time()
        scrapes = [
            ScrapeResult(
                text=text,
                timestamp=start + idx * 10.0,
                ok=True,
                status_code="self-test",
                duration_seconds=0.0,
                response_bytes=len(text.encode("utf-8")),
            )
            for idx, text in enumerate(self_test_texts())
        ]
        collect_from_scrapes(scrapes, args, build_label_filters(args))

        summary_path = Path(tmp) / "summary.csv"
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            rows = {row["metric"]: row for row in csv.DictReader(f)}

        expected = {
            "e2e_request_latency_avg_s": 2.0,
            "ttft_avg_s": 0.3,
            "tpot_avg_s": 0.2,
            "prompt_tokens_per_s": 6.0,
            "generation_tokens_per_s": 10.0,
            "prefix_cache_hit_rate": 0.6,
            "external_prefix_cache_hit_rate": 0.4,
            "posix_store_load_ratio": 0.5,
            "posix_store_load_fraction": 0.5,
            "cache_store_load_hit_fraction": 0.5,
        }
        for metric, expected_value in expected.items():
            assert_close(float(rows[metric]["value"]), expected_value, metric)

        timeseries_path = Path(tmp) / "timeseries.csv"
        with timeseries_path.open("r", encoding="utf-8", newline="") as f:
            timeseries_rows = list(csv.DictReader(f))
        if len(timeseries_rows) != 1:
            raise AssertionError(f"expected 1 interval row, got {len(timeseries_rows)}")
        plot_dir = Path(tmp) / "plots"
        expected_plots = [
            "e2e_request_latency_avg",
            "ttft_avg",
            "tpot_avg",
            "prompt_token_throughput",
            "generation_token_throughput",
            "cache_hit_rates",
            "cache_store_vs_posix_store_load",
        ]
        missing_plots = [
            name
            for name in expected_plots
            if not (plot_dir / f"{name}.png").exists()
            and not (plot_dir / f"{name}.svg").exists()
        ]
        if missing_plots:
            raise AssertionError(f"missing expected plot(s): {', '.join(missing_plots)}")

    print("[INFO] self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lightweight Grafana-like collector for selected vLLM/UCM metrics."
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Prometheus metrics endpoint, e.g. http://127.0.0.1:8000/metrics",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="scrape interval in seconds",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="total collection duration in seconds. If 0, run until stopped.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--out",
        default="./vllm_metrics_result",
        help="output directory",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="exact model_name label filter",
    )
    parser.add_argument(
        "--job-regex",
        default=None,
        help="regex filter for the job label",
    )
    parser.add_argument(
        "--worker-id-regex",
        default=None,
        help="regex filter for the worker_id label",
    )
    parser.add_argument(
        "--instance-regex",
        default=None,
        help="regex filter for the instance label",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="extra exact label filter in KEY=VALUE form. Can be repeated.",
    )
    parser.add_argument(
        "--labels-contains",
        default=None,
        help="optional substring filter applied to normalized labels JSON",
    )
    parser.add_argument(
        "--stop-file",
        default=None,
        help="optional file path used to stop collection gracefully",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="scrape only once. At least two scrapes are needed for interval metrics.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue scraping when one HTTP request fails",
    )
    parser.add_argument(
        "--replay-files",
        nargs="+",
        default=None,
        help="read one or more Prometheus text files as simulated scrapes",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run built-in formula checks with simulated metrics",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="generate PNG plots from timeseries.csv after collection",
    )
    parser.add_argument(
        "--plot-only",
        default=None,
        metavar="OUT_DIR",
        help="generate plots from an existing output directory and exit",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.plot_only:
        generate_plots(Path(args.plot_only))
        return

    print("[INFO] lightweight metrics collector starting")
    print(f"[INFO] output dir: {args.out}")
    if args.replay_files:
        print(f"[INFO] replay files: {len(args.replay_files)}")
    else:
        print(f"[INFO] metrics url: {args.url}")
        print(f"[INFO] interval: {args.interval}s")
        print(f"[INFO] duration: {args.duration}s")

    run_collection(args)


if __name__ == "__main__":
    main()
