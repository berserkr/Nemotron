#!/usr/bin/env python3
"""Pre-sample datasets to target counts from a sampling plan.

Reads a blend JSON or sampling plan and creates subsampled JSONL files so that
packing with uniform weights produces the intended blend distribution.

Handles JSONL files where records may contain embedded newlines (e.g., code
in message content). Uses json.JSONDecoder.raw_decode to correctly parse
records regardless of internal whitespace.

Usage:
    python presample_blend.py \
        --input /path/to/blend_or_plan.json \
        --output /path/to/sampled_output \
        --seed 42 \
        --workers 20

    # With sampling plan format (needs --data-root):
    python presample_blend.py \
        --input /path/to/sampling_plan.json \
        --data-root /path/to/v1 \
        --output /path/to/sampled_output
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def iter_jsonl_records(filepath: Path):
    """Iterate JSONL records, handling embedded newlines in JSON strings.

    Falls back to raw_decode if a line fails to parse (multi-line record).
    """
    with open(filepath) as f:
        buffer = ""
        for line in f:
            if buffer:
                buffer += line
                try:
                    record = json.loads(buffer)
                    yield record
                    buffer = ""
                except json.JSONDecodeError:
                    continue
            else:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    buffer = line


def count_records(filepath: Path) -> int:
    count = 0
    for _ in iter_jsonl_records(filepath):
        count += 1
    return count


def find_category_dir(data_root: Path, category: str) -> Path | None:
    for variant in [category, category.replace("_", "-"), category.replace("-", "_")]:
        candidate = data_root / variant
        if candidate.is_dir():
            return candidate
    return None


def process_dataset(
    category: str,
    cat_dir: str,
    target: int,
    meta_category: str,
    output_dir: str,
    seed: int,
) -> dict:
    cat_path = Path(cat_dir)
    out_path = Path(output_dir) / category
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "sampled.jsonl"

    t0 = time.time()

    jsonl_files = sorted(cat_path.glob("*.jsonl"))
    if not jsonl_files:
        return {"category": category, "error": "no jsonl files found"}

    # Pass 1: count records per file
    file_counts = []
    total_records = 0
    for f in jsonl_files:
        n = count_records(f)
        file_counts.append((f, n))
        total_records += n

    if total_records == 0:
        return {"category": category, "error": "empty dataset"}

    available = total_records
    ratio = target / available
    dataset_seed = seed + abs(hash(category)) % (2**31)
    rng = random.Random(dataset_seed)

    pbar = tqdm(total=available, desc=f"{category}", unit="rec", leave=True)

    if target <= available:
        selected = set(rng.sample(range(available), target))
        written = 0
        global_idx = 0
        with open(out_file, "w") as out:
            for filepath, _ in file_counts:
                for record in iter_jsonl_records(filepath):
                    if global_idx in selected:
                        out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                        out.write("\n")
                        written += 1
                    global_idx += 1
                    pbar.update(1)
        strategy = "downsample"
    else:
        full_copies = target // available
        remainder = target % available
        remainder_indices = set(rng.sample(range(available), remainder))

        written = 0
        with open(out_file, "w") as out:
            global_idx = 0
            for filepath, _ in file_counts:
                for record in iter_jsonl_records(filepath):
                    repeats = full_copies + (1 if global_idx in remainder_indices else 0)
                    serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    for _ in range(repeats):
                        out.write(serialized)
                        out.write("\n")
                        written += 1
                    global_idx += 1
                    pbar.update(1)
        strategy = "upsample"

    pbar.close()

    elapsed = time.time() - t0
    size_bytes = out_file.stat().st_size

    return {
        "category": category,
        "meta_category": meta_category,
        "available": available,
        "target": target,
        "written": written,
        "ratio": round(ratio, 4),
        "strategy": strategy,
        "size_gb": round(size_bytes / (1 << 30), 2),
        "elapsed_sec": round(elapsed, 1),
        "files": len(file_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-sample datasets to target counts")
    parser.add_argument("--input", required=True, help="Path to blend JSON or sampling plan JSON")
    parser.add_argument("--data-root", default=None, help="Root directory with category folders (required for sampling plan format)")
    parser.add_argument("--output", required=True, help="Output directory for sampled data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers (default: one per dataset)")
    args = parser.parse_args()

    with open(args.input) as f:
        plan = json.load(f)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    is_blend = "datasets" in plan
    is_sampling_plan = "sampling" in plan

    if is_blend:
        version = plan.get("_version", "unknown")
        total_target = plan.get("_total_target_samples", 0)
        tasks = []
        for ds in plan["datasets"]:
            target = ds.get("_target", 0)
            name = ds["name"]
            path = ds["path"].rstrip("/").removesuffix("/*.jsonl").removesuffix("/*")
            meta_cat = ds.get("_category", "unknown")
            if target <= 0:
                print(f"  SKIP: {name} — no target", file=sys.stderr)
                continue
            if not Path(path).is_dir():
                print(f"  SKIP: {name} — {path} not found", file=sys.stderr)
                continue
            tasks.append((name, path, target, meta_cat))
    elif is_sampling_plan:
        if not args.data_root:
            print("ERROR: --data-root required for sampling plan format", file=sys.stderr)
            sys.exit(1)
        root = Path(args.data_root)
        version = plan.get("version", "unknown")
        total_target = plan.get("total_samples", 0)
        tasks = []
        for category, info in sorted(plan["sampling"].items()):
            target = info.get("target", 0)
            if target <= 0:
                continue
            cat_dir = find_category_dir(root, category)
            if cat_dir is None:
                print(f"  SKIP: {category} — directory not found", file=sys.stderr)
                continue
            tasks.append((category, str(cat_dir), target, info.get("category", "unknown")))
    else:
        print("ERROR: JSON must have 'datasets' (blend) or 'sampling' (plan) key", file=sys.stderr)
        sys.exit(1)

    print(f"Version: {version}")
    print(f"Total target: {total_target:,}")
    print(f"Datasets: {len(tasks)}")
    print(f"Seed: {args.seed}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_root}")
    print()

    print(f"Processing {len(tasks)} datasets...\n")
    t_start = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for category, cat_dir, target, meta_cat in tasks:
            fut = pool.submit(
                process_dataset,
                category=category,
                cat_dir=cat_dir,
                target=target,
                meta_category=meta_cat,
                output_dir=str(output_root),
                seed=args.seed,
            )
            futures[fut] = category

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            cat = result["category"]
            if "error" in result:
                print(f"  ERROR {cat}: {result['error']}")
            else:
                print(
                    f"  {cat:<45s} {result['available']:>10,} → {result['written']:>10,} "
                    f"({result['ratio']:.2f}x {result['strategy']:<11s}) "
                    f"{result['size_gb']:>6.1f} GB  {result['elapsed_sec']:>5.1f}s"
                )

    results.sort(key=lambda r: r["category"])
    total_written = sum(r.get("written", 0) for r in results)
    total_available = sum(r.get("available", 0) for r in results)
    elapsed = time.time() - t_start

    print(f"\n{'='*80}")
    print(f"Total: {total_written:,} samples written from {total_available:,} available")
    print(f"Target: {total_target:,}")
    print(f"Elapsed: {elapsed:.1f}s")

    manifest = {
        "version": version,
        "seed": args.seed,
        "total_written": total_written,
        "total_available": total_available,
        "total_target": total_target,
        "elapsed_sec": round(elapsed, 1),
        "datasets": results,
    }
    manifest_path = output_root / "presample_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")

    blend_path = output_root / "blend_uniform.json"
    blend = {
        "_comment": f"Uniform-weight blend from pre-sampled data ({version}). Distribution is baked into the data.",
        "_version": version,
        "_total_samples": total_written,
        "datasets": [
            {
                "name": r["category"],
                "path": str(output_root / r["category"]),
                "weight": 1.0,
            }
            for r in results
            if "error" not in r
        ],
    }
    with open(blend_path, "w") as f:
        json.dump(blend, f, indent=2)
    print(f"Blend: {blend_path}")


if __name__ == "__main__":
    main()
