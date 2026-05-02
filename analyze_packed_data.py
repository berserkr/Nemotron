#!/usr/bin/env python3
"""Analyze packed SFT training data: token counts, packing stats, training schedule.

Usage:
    python analyze_packed_data.py /path/to/splits/train
    python analyze_packed_data.py /path/to/splits/train --gbs 128 --pack-size 65536
    python analyze_packed_data.py /path/to/splits/train --full  # read all shards
    python analyze_packed_data.py /path/to/splits/train --workers 32
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _read_metadata(path_str: str) -> int:
    return pq.read_metadata(path_str).num_rows


def _process_shard(path_str: str) -> tuple[list[int], list[int], list[int]]:
    t = pq.read_table(path_str, columns=["input_ids", "seq_start_id"])
    col_ids = t.column("input_ids")
    col_starts = t.column("seq_start_id")

    bin_lengths = []
    doc_lengths = []
    seq_counts = []

    for i in range(t.num_rows):
        ids = col_ids[i].as_py()
        starts = col_starts[i].as_py()
        bin_len = len(ids)
        bin_lengths.append(bin_len)
        seq_counts.append(len(starts))

        boundaries = starts + [bin_len]
        for j in range(len(boundaries) - 1):
            doc_lengths.append(boundaries[j + 1] - boundaries[j])

    return bin_lengths, doc_lengths, seq_counts


def analyze(train_dir: Path, gbs: int, pack_size: int, full: bool, workers: int):
    files = sorted(train_dir.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {train_dir}")
        sys.exit(1)

    file_strs = [str(f) for f in files]

    # Total rows from metadata (parallel, reads only file footers)
    print(f"Reading metadata from {len(files):,} files ({workers} workers)...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        total_rows = sum(pool.map(_read_metadata, file_strs, chunksize=64))

    # Sample shards for detailed stats
    if full:
        sample_files = file_strs
    else:
        sample_files = file_strs[:: max(1, len(file_strs) // 100)]

    print(f"Analyzing {len(sample_files)} shards ({workers} workers)...", flush=True)

    all_bin_lengths = []
    all_doc_lengths = []
    all_seq_counts = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_shard, f): f for f in sample_files}
        done = 0
        for future in as_completed(futures):
            bl, dl, sc = future.result()
            all_bin_lengths.extend(bl)
            all_doc_lengths.extend(dl)
            all_seq_counts.extend(sc)
            done += 1
            if done % 50 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} shards processed", flush=True)

    bin_lengths = np.array(all_bin_lengths)
    doc_lengths = np.array(all_doc_lengths)
    seq_counts = np.array(all_seq_counts)

    total_tokens_approx = total_rows * bin_lengths.mean()
    iters_per_epoch = total_rows // gbs

    print()
    print(f"{'=' * 60}")
    print(f"Packed Data Analysis")
    print(f"{'=' * 60}")
    print(f"Path:       {train_dir}")
    print(f"Shards:     {len(files):,} files ({len(sample_files)} sampled)")
    print()

    print(f"--- Overview ---")
    print(f"  Total packed sequences: {total_rows:,}")
    print(f"  Total tokens (approx):  {total_tokens_approx / 1e9:.1f}B")
    print(f"  Total documents (est):  {total_rows * seq_counts.mean():,.0f}")
    print()

    print(f"--- Packed Bin Lengths ---")
    print(f"  Mean:       {bin_lengths.mean():,.0f} tokens")
    print(f"  Median:     {np.median(bin_lengths):,.0f} tokens")
    print(f"  Min:        {bin_lengths.min():,} tokens")
    print(f"  Max:        {bin_lengths.max():,} tokens")
    print(f"  Std:        {bin_lengths.std():,.0f} tokens")
    print(f"  Efficiency: {bin_lengths.mean() / pack_size * 100:.1f}% of {pack_size:,}")
    print()

    print(f"--- Documents Per Bin ---")
    print(f"  Mean:   {seq_counts.mean():.1f}")
    print(f"  Median: {np.median(seq_counts):.0f}")
    print(f"  Min:    {seq_counts.min()}")
    print(f"  Max:    {seq_counts.max()}")
    print()

    print(f"--- Per-Document Token Lengths ---")
    print(f"  Sampled:  {len(doc_lengths):,} documents")
    print(f"  Mean:     {doc_lengths.mean():,.0f} tokens")
    print(f"  Median:   {np.median(doc_lengths):,.0f} tokens")
    print(f"  Min:      {doc_lengths.min():,} tokens")
    print(f"  Max:      {doc_lengths.max():,} tokens")
    print(f"  Std:      {doc_lengths.std():,.0f} tokens")
    print(f"  p10:      {np.percentile(doc_lengths, 10):,.0f} tokens")
    print(f"  p25:      {np.percentile(doc_lengths, 25):,.0f} tokens")
    print(f"  p75:      {np.percentile(doc_lengths, 75):,.0f} tokens")
    print(f"  p90:      {np.percentile(doc_lengths, 90):,.0f} tokens")
    print(f"  p99:      {np.percentile(doc_lengths, 99):,.0f} tokens")
    print()

    print(f"--- Training Schedule (GBS={gbs}) ---")
    print(f"  Iters/epoch:    {iters_per_epoch:,}")
    print(f"  2 epochs:       {2 * iters_per_epoch:,}")
    print(f"  3 epochs:       {3 * iters_per_epoch:,}")
    print(f"  Warmup (5%):    {int(0.05 * iters_per_epoch):,}")
    print(f"  Warmup (1%):    {int(0.01 * iters_per_epoch):,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze packed SFT training data")
    parser.add_argument("train_dir", type=Path, help="Path to splits/train directory")
    parser.add_argument("--gbs", type=int, default=64, help="Global batch size (default: 64)")
    parser.add_argument("--pack-size", type=int, default=131072, help="Pack size in tokens (default: 131072)")
    parser.add_argument("--full", action="store_true", help="Read all shards (exact, uses parallel workers)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 16, help="Parallel workers (default: all CPUs)")
    args = parser.parse_args()

    analyze(args.train_dir, args.gbs, args.pack_size, args.full, args.workers)
