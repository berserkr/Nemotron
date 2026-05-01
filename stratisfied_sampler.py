#!/usr/bin/env python3
"""Stratified pre-sampling — numpy index, streaming write, minimal memory.

Two-pass approach:
  Pass 1: Scan files in parallel, collect (file_idx, byte_offset, line_len,
           total_chars, thinking_chars) into a numpy array (~40 bytes/line).
  Pass 2: Compute rates, select top-N by adjusted random key, stream raw
           bytes from original files to output (zero JSON in memory).

Memory: ~400MB for 10M lines (numpy int64 array) vs ~8GB with Python tuples.

Usage:
    python stratified_presample.py \
        --input /path/to/sampling_7m_balanced.json \
        --data-root /path/to/v1 \
        --output /path/to/v1_sampled_7m_stratified \
        --workers 64 \
        --seed 42


For 128k context (131,072 tokens ≈ 524,000 chars), you want conversations to fit in a single packed bin. Anything longer gets truncated by the packer and loses <|im_end|>. Adjust buckets to aggressively cut near and above the pack boundary:

  Length buckets for 128k packing:
  [[0, 8000, 1.0], [8000, 32000, 1.0], [32000, 128000, 0.8], [128000, 384000, 0.5], [384000, 524000, 0.2], [524000, -1, 0.05]]

  ┌───────────────┬─────────┬───────────┬──────────────────────────────┐
  │ Range (chars) │ ~Tokens │ Keep rate │             Why              │
  ├───────────────┼─────────┼───────────┼──────────────────────────────┤
  │ 0-8k          │ 0-2k    │ 100%      │ Short, great for stop signal │
  ├───────────────┼─────────┼───────────┼──────────────────────────────┤
  │ 8k-32k        │ 2-8k    │ 100%      │ Normal conversations         │
  ├───────────────┼─────────┼───────────┼──────────────────────────────┤
  │ 32k-128k      │ 8-32k   │ 80%       │ Good length, keep most       │
  ├───────────────┼─────────┼───────────┼──────────────────────────────┤
  │ 128k-384k     │ 32-96k  │ 50%       │ Takes 25-75% of a bin        │
  ├───────────────┼─────────┼───────────┼──────────────────────────────┤
  │ 384k-524k     │ 96-128k │ 20%       │ Nearly fills a bin alone     │
  ├───────────────┼─────────┼───────────┼──────────────────────────────┤
  │ 524k+         │ 128k+   │ 5%        │ Will be truncated — keep few │
  └───────────────┴─────────┴───────────┴──────────────────────────────┘

  Thinking buckets (same as before, thinking length is independent of pack size):
  [[0, 0, 1.0], [1, 2000, 1.0], [2000, 8000, 0.8], [8000, 32000, 0.5], [32000, -1, 0.2]]

  Usage:

  python scripts/stratified_presample.py \
      --input /path/to/sampling_7m_balanced.json \
      --data-root /mnt/vast/proj/checkpoints/bathen/datasets/sft/granite-4.2-sft-datasets-r260414a/v1 \
      --output /mnt/vast/proj/checkpoints/bathen/datasets/sft/granite-4.2-sft-datasets-r260414a/v1_sampled_7m_stratified \
      --workers 128 \
      --seed 42 \
      --length-buckets '[[0,8000,1.0],[8000,32000,1.0],[32000,128000,0.8],[128000,384000,0.5],[384000,524000,0.2],[524000,-1,0.05]]'

  This ensures most bins pack multiple conversations (better stop signal ratio) and almost nothing gets truncated at the 128k boundary.
  
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

LENGTH_BUCKETS = [
    (0, 8000, 1.0),
    (8000, 32000, 1.0),
    (32000, 128000, 0.8),
    (128000, 384000, 0.5),
    (384000, 524000, 0.2),
    (524000, -1, 0.05),
]

THINKING_BUCKETS = [
    (0, 0, 1.0),
    (1, 2000, 1.0),
    (2000, 8000, 0.8),
    (8000, 32000, 0.5),
    (32000, -1, 0.2),
]


def get_rate_vec(values, buckets):
    """Vectorized bucket rate lookup. Returns float64 array of rates."""
    rates = np.ones(len(values), dtype=np.float64)
    for bmin, bmax, rate in buckets:
        upper = np.iinfo(np.int64).max if bmax == -1 else bmax
        mask = (values >= bmin) & (values <= upper)
        rates[mask] = rate
    return rates


def bucket_label(value, buckets):
    for bmin, bmax, rate in buckets:
        upper = float("inf") if bmax == -1 else bmax
        if bmin <= value <= upper:
            return f"{bmin}-{bmax}" if bmax != -1 else f"{bmin}+"
    return "?"


def scan_chunk(args):
    """Scan a byte range. Returns flat list: [file_idx, offset, line_len, total_chars, thinking_chars, ...]"""
    path, start, end, file_idx = args
    # Pre-allocate list, will be flattened to numpy later
    data = []
    with open(path, "rb") as fb:
        fb.seek(start)
        pos = start
        while pos < end:
            raw = fb.readline()
            if not raw:
                break
            line_len = len(raw)
            next_pos = pos + line_len

            stripped = raw.strip()
            if stripped:
                total_c = 0
                think_c = 0
                try:
                    obj = json.loads(stripped)
                    messages = obj.get("messages", [])
                    for m in messages:
                        if not isinstance(m, dict):
                            continue
                        c = m.get("content")
                        if isinstance(c, str):
                            total_c += len(c)
                            if "<think>" in c and "</think>" in c:
                                s = c.find("<think>") + 7
                                e = c.find("</think>")
                                if e > s:
                                    think_c += e - s
                        rc = m.get("reasoning_content")
                        if isinstance(rc, str):
                            total_c += len(rc)
                            think_c += len(rc)
                except (json.JSONDecodeError, Exception):
                    pass
                # Append as flat values (5 per line)
                data.extend((file_idx, pos, line_len, total_c, think_c))

            pos = next_pos
    return data


def find_line_offsets(path, num_chunks):
    fsize = os.path.getsize(path)
    if fsize == 0:
        return []
    chunk_size = fsize // num_chunks
    offsets = []
    with open(path, "rb") as f:
        start = 0
        for _ in range(num_chunks - 1):
            f.seek(min(start + chunk_size, fsize))
            f.readline()
            end = f.tell()
            if end >= fsize:
                offsets.append((start, fsize))
                return offsets
            offsets.append((start, end))
            start = end
        offsets.append((start, fsize))
    return offsets


def find_category_dir(data_root, category):
    for variant in [category, category.replace("_", "-"), category.replace("-", "_")]:
        candidate = data_root / variant
        if candidate.is_dir():
            return candidate
    return None


def process_category(
    category, cat_dir, target, seed, workers, length_buckets, thinking_buckets, output_dir, dry_run,
):
    t0 = time.time()
    cat_path = Path(cat_dir)
    jsonl_files = sorted(str(f) for f in cat_path.glob("*.jsonl"))
    if not jsonl_files:
        return {"category": category, "error": "no jsonl files found"}

    # Build chunks
    CHUNK_TARGET = 512 << 20
    all_chunks = []
    total_bytes = 0
    for fi, f in enumerate(jsonl_files):
        fsize = os.path.getsize(f)
        total_bytes += fsize
        nc = max(1, fsize // CHUNK_TARGET)
        nc = min(nc, workers)
        for start, end in find_line_offsets(f, nc):
            all_chunks.append((f, start, end, fi))

    # Pass 1: Scan — collect flat int arrays from workers
    flat_data = []
    actual_workers = min(len(all_chunks), workers)

    with ProcessPoolExecutor(max_workers=actual_workers) as pool:
        futures = {pool.submit(scan_chunk, chunk): chunk for chunk in all_chunks}
        with tqdm(total=total_bytes, unit="B", unit_scale=True,
                  desc=f"  Scan {category}", file=sys.stderr, leave=False) as pbar:
            for future in as_completed(futures):
                chunk = futures[future]
                chunk_bytes = chunk[2] - chunk[1]
                result = future.result()
                flat_data.extend(result)
                pbar.update(chunk_bytes)
                del result  # free worker result immediately

    if not flat_data:
        return {"category": category, "error": "empty"}

    # Convert to numpy — reshape from flat to (N, 5)
    arr = np.array(flat_data, dtype=np.int64).reshape(-1, 5)
    del flat_data  # free the Python list

    available = len(arr)
    # Columns: 0=file_idx, 1=offset, 2=line_len, 3=total_chars, 4=thinking_chars

    # Vectorized rate computation
    length_rates = get_rate_vec(arr[:, 3], length_buckets)
    thinking_rates = get_rate_vec(arr[:, 4], thinking_buckets)
    effective_rates = np.minimum(length_rates, thinking_rates)
    del length_rates, thinking_rates

    # Random keys
    rng = np.random.RandomState(seed + abs(hash(category)) % (2**31))
    rand_keys = rng.random(available)

    # Compute bucket labels for stats (Python loop but only for display)
    bucket_stats = defaultdict(lambda: {"total": 0, "kept": 0})
    bl_arr = arr[:, 3]  # total_chars
    bt_arr = arr[:, 4]  # thinking_chars

    # Vectorized bucket assignment for stats
    for i in range(available):
        ll = bucket_label(int(bl_arr[i]), length_buckets)
        tl = bucket_label(int(bt_arr[i]), thinking_buckets)
        bk = f"len={ll} think={tl}"
        bucket_stats[bk]["total"] += 1

    if target <= available:
        # Downsample: scale rates to hit target
        expected = float(effective_rates.sum())
        scale = target / expected if expected > 0 else 1.0

        # Select where rand_key < rate * scale
        adjusted = np.minimum(effective_rates * scale, 1.0)
        mask = rand_keys < adjusted
        selected_indices = np.where(mask)[0]

        # Trim overshoot
        if len(selected_indices) > target:
            rng.shuffle(selected_indices)
            selected_indices = selected_indices[:target]

        # Fill undershoot
        if len(selected_indices) < target:
            not_selected = np.where(~mask)[0]
            rng.shuffle(not_selected)
            deficit = target - len(selected_indices)
            selected_indices = np.concatenate([selected_indices, not_selected[:deficit]])

        selected = arr[selected_indices]
    else:
        # Upsample: take all + repeat
        full_copies = target // available
        remainder = target % available
        indices = np.tile(np.arange(available), full_copies)
        if remainder > 0:
            extra = rng.choice(available, remainder, replace=False)
            indices = np.concatenate([indices, extra])
        selected = arr[indices]

    num_selected = len(selected)

    # Bucket stats for selected
    for i in range(num_selected):
        ll = bucket_label(int(selected[i, 3]), length_buckets)
        tl = bucket_label(int(selected[i, 4]), thinking_buckets)
        bk = f"len={ll} think={tl}"
        bucket_stats[bk]["kept"] += 1

    # Pass 2: Stream selected lines to output
    size_gb = 0
    out_file_str = None
    if not dry_run and num_selected > 0:
        cat_out = Path(output_dir) / category
        cat_out.mkdir(parents=True, exist_ok=True)
        out_file = cat_out / "sampled.jsonl"
        out_file_str = str(out_file)

        # Sort by (file_idx, offset) for sequential disk reads
        sort_order = np.lexsort((selected[:, 1], selected[:, 0]))
        selected_sorted = selected[sort_order]

        written = 0
        with open(out_file_str, "wb", buffering=4 << 20) as fout:
            current_fi = -1
            fin = None
            for i in range(len(selected_sorted)):
                fi = int(selected_sorted[i, 0])
                offset = int(selected_sorted[i, 1])
                line_len = int(selected_sorted[i, 2])

                if fi != current_fi:
                    if fin is not None:
                        fin.close()
                    fin = open(jsonl_files[fi], "rb")
                    current_fi = fi

                fin.seek(offset)
                raw = fin.read(line_len)
                if not raw.endswith(b"\n"):
                    raw += b"\n"
                fout.write(raw)
                written += 1

            if fin is not None:
                fin.close()

        size_gb = os.path.getsize(out_file_str) / (1 << 30)

    elapsed = time.time() - t0

    # Free numpy arrays
    del arr, selected, effective_rates, rand_keys

    return {
        "category": category,
        "available": available,
        "target": target,
        "selected": num_selected,
        "buckets": dict(bucket_stats),
        "elapsed_sec": round(elapsed, 1),
        "total_bytes": total_bytes,
        "size_gb": round(size_gb, 2),
        "out_file": out_file_str,
    }


def main():
    parser = argparse.ArgumentParser(description="Stratified pre-sampling (numpy, streaming write)")
    parser.add_argument("--input", required=True, help="Sampling plan JSON")
    parser.add_argument("--data-root", required=True, help="Root with category folders")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--length-buckets", help="JSON: [[min,max,rate],...]")
    parser.add_argument("--thinking-buckets", help="JSON: [[min,max,rate],...]")
    args = parser.parse_args()

    with open(args.input) as f:
        plan = json.load(f)

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    lb = LENGTH_BUCKETS
    tb = THINKING_BUCKETS
    if args.length_buckets:
        lb = [tuple(b) for b in json.loads(args.length_buckets)]
    if args.thinking_buckets:
        tb = [tuple(b) for b in json.loads(args.thinking_buckets)]

    sampling = plan.get("sampling", plan.get("categories", {}))

    print(f"Plan:        {args.input}", file=sys.stderr)
    print(f"Data root:   {data_root}", file=sys.stderr)
    print(f"Output:      {output_dir}", file=sys.stderr)
    print(f"Categories:  {len(sampling)}", file=sys.stderr)
    print(f"Workers:     {args.workers}", file=sys.stderr)
    print(f"Length:      {lb}", file=sys.stderr)
    print(f"Thinking:    {tb}", file=sys.stderr)
    print(file=sys.stderr)

    grand = {"available": 0, "target": 0, "selected": 0}

    for cat_name in sorted(sampling.keys()):
        cat_cfg = sampling[cat_name]
        target = cat_cfg["target"]

        cat_dir = find_category_dir(data_root, cat_name)
        if cat_dir is None:
            print(f"  SKIP {cat_name}: not found", file=sys.stderr)
            continue

        print(f"\n{'='*70}", file=sys.stderr)
        print(f"  {cat_name} (target={target:,})", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

        result = process_category(
            cat_name, str(cat_dir), target, args.seed, args.workers,
            lb, tb, str(output_dir), args.dry_run,
        )

        if "error" in result:
            print(f"  ERROR: {result['error']}", file=sys.stderr)
            continue

        print(f"  Available: {result['available']:,}", file=sys.stderr)
        print(f"  Target:    {result['target']:,}", file=sys.stderr)
        print(f"  Selected:  {result['selected']:,}", file=sys.stderr)
        print(f"  Time:      {result['elapsed_sec']}s", file=sys.stderr)
        if result.get("size_gb"):
            print(f"  Size:      {result['size_gb']} GB", file=sys.stderr)

        if result.get("buckets"):
            print(f"  Buckets:", file=sys.stderr)
            for bk, bv in sorted(result["buckets"].items()):
                pct = bv['kept'] / max(bv['total'], 1) * 100
                print(f"    {bk:<45s} {bv['kept']:>8,}/{bv['total']:>8,} ({pct:>5.1f}%)", file=sys.stderr)

        if result.get("out_file"):
            print(f"  Written:   {result['out_file']}", file=sys.stderr)

        grand["available"] += result["available"]
        grand["target"] += result["target"]
        grand["selected"] += result["selected"]

    print(f"\n{'='*70}", file=sys.stderr)
    print(f"TOTAL: {grand['available']:,} avail, {grand['target']:,} target, {grand['selected']:,} selected",
          file=sys.stderr)


if __name__ == "__main__":
    main()
