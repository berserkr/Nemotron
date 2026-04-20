#!/usr/bin/env python3
"""Convert a sampling plan JSON to a Nemotron packing blend JSON.

Maps category names to JSONL files on disk and distributes weights
proportional to file sizes within each category.

Usage:
    python sampling_to_blend.py \
        --sampling /path/to/sampling_plan.json \
        --data-root /proj/datasets/sft_datasets/granite-4.2-sft-datasets-r260414a/v1 \
        --output /path/to/blend.json

The script:
1. Reads the sampling plan (with target counts per category)
2. Discovers JSONL files under data-root for each category
3. Distributes the category's target weight across files proportional to file size
4. Writes a Nemotron-compatible blend JSON
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _count_lines(filepath: Path) -> int:
    """Count lines in a file efficiently using buffered read."""
    count = 0
    with open(filepath, "rb") as f:
        buf = f.raw.read(1 << 20)
        while buf:
            count += buf.count(b"\n")
            buf = f.raw.read(1 << 20)
    return count


def discover_files(data_root: Path, category: str, count_samples: bool = True) -> list[dict]:
    """Find all JSONL files for a category under data_root.

    Returns list of {"path": str, "size": int, "samples": int, "name": str}.
    When count_samples=True, counts lines (=samples) per file for accurate
    weight distribution. Falls back to file size if counting fails.
    """
    category_dir = data_root / category
    if not category_dir.is_dir():
        # Try common variations
        for variant in [
            category,
            category.replace("_", "-"),
            category.replace("-", "_"),
        ]:
            candidate = data_root / variant
            if candidate.is_dir():
                category_dir = candidate
                break
        else:
            return []

    files = []
    for f in sorted(category_dir.glob("*.jsonl")):
        size = f.stat().st_size
        samples = 0
        if count_samples:
            try:
                samples = _count_lines(f)
                print(f"    {f.name}: {samples:,} samples ({size / (1 << 30):.1f} GB)")
            except Exception:
                samples = 0
        files.append({
            "path": f"file://{f}",
            "size": size,
            "samples": samples,
            "name": f.stem,
        })
    return files


def make_dataset_name(category: str, filename: str) -> str:
    """Create a short, unique dataset name from category and filename."""
    # Shorten common prefixes
    name = filename
    for prefix in ["Nemotron-", "Nemotron_", "nemotron-", "nemotron_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Abbreviate category
    cat_short = category.replace("_", "-")
    if len(cat_short) > 20:
        parts = cat_short.split("-")
        cat_short = "-".join(p[:4] for p in parts)

    return f"{cat_short}/{name}"[:80]


def sampling_to_blend(sampling_path: str, data_root: str, output_path: str):
    """Convert sampling plan to blend JSON."""
    with open(sampling_path) as f:
        plan = json.load(f)

    root = Path(data_root)
    version = plan.get("version", "unknown")
    total_target = plan.get("total_samples", 0)
    sampling = plan.get("sampling", {})

    datasets = []
    stats = {
        "categories_found": 0,
        "categories_missing": [],
        "total_files": 0,
        "total_weight": 0,
    }

    for category, info in sorted(sampling.items()):
        target = info.get("target", 0)
        strategy = info.get("strategy", "unknown")
        ratio = info.get("ratio", 1.0)
        meta_category = info.get("category", "unknown")

        files = discover_files(root, category)

        if not files:
            stats["categories_missing"].append(category)
            print(f"  WARNING: No files found for category '{category}' under {root}", file=sys.stderr)
            continue

        stats["categories_found"] += 1
        stats["total_files"] += len(files)

        # Distribute weight proportional to sample count (fallback to file size)
        total_samples_in_cat = sum(f["samples"] for f in files)
        use_samples = total_samples_in_cat > 0
        if use_samples:
            total_for_fraction = total_samples_in_cat
        else:
            total_for_fraction = sum(f["size"] for f in files) or 1

        # Scale: target / 1000 for readable weights
        category_weight = target / 1000.0

        for i, file_info in enumerate(files):
            if use_samples:
                fraction = file_info["samples"] / total_for_fraction
            else:
                fraction = file_info["size"] / total_for_fraction
            weight = round(category_weight * fraction, 1)
            if weight < 0.1:
                weight = 0.1  # minimum weight to ensure inclusion

            entry = {
                "name": make_dataset_name(category, file_info["name"]),
                "path": file_info["path"],
                "weight": weight,
            }

            # Add metadata as comments on first entry per category
            if i == 0:
                entry["_category"] = category
                entry["_meta_category"] = meta_category
                entry["_strategy"] = f"{strategy} ({ratio}x)"
                entry["_target"] = target
                entry["_available"] = info.get("available", "?")
                entry["_actual_samples"] = total_samples_in_cat
                entry["_files_in_category"] = len(files)
                entry["_weight_method"] = "sample_count" if use_samples else "file_size"

            datasets.append(entry)
            stats["total_weight"] += weight

    # Build blend JSON
    blend = {
        "_comment": f"Auto-generated blend from sampling plan: {version}. "
                    f"Target: {total_target:,} samples. "
                    f"Categories: {stats['categories_found']} found, "
                    f"{len(stats['categories_missing'])} missing.",
        "_version": version,
        "_total_target_samples": total_target,
        "_category_distribution": plan.get("category_distribution", {}),
    }

    if stats["categories_missing"]:
        blend["_missing_categories"] = stats["categories_missing"]

    blend["datasets"] = datasets

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(blend, f, indent=2)

    # Print summary
    print(f"\nBlend generation summary:")
    print(f"  Sampling plan: {version}")
    print(f"  Target samples: {total_target:,}")
    print(f"  Categories found: {stats['categories_found']}")
    print(f"  Categories missing: {len(stats['categories_missing'])}")
    if stats["categories_missing"]:
        for c in stats["categories_missing"]:
            print(f"    - {c}")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total weight: {stats['total_weight']:.1f}")
    print(f"  Output: {output}")

    # Print per-category breakdown
    print(f"\n  {'Category':<40s} {'Target':>10s} {'Actual':>10s} {'Files':>6s} {'Weight':>10s} {'Method':<12s}")
    print(f"  {'-'*90}")
    for category, info in sorted(sampling.items()):
        target = info.get("target", 0)
        files = discover_files(root, category, count_samples=False)  # don't recount
        actual = sum(f.get("samples", 0) for f in files)
        cat_entries = [d for d in datasets if d.get("_category") == category]
        cat_weight = sum(
            d["weight"] for d in datasets
            if d.get("_category") == category or
            (not d.get("_category") and category.replace("_", "-") in d.get("name", ""))
        )
        method = cat_entries[0].get("_weight_method", "?") if cat_entries else "?"
        status = "OK" if files else "MISSING"
        print(f"  {category:<40s} {target:>10,} {actual:>10,} {len(files):>6} {cat_weight:>10.1f}  {method:<12s} {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert sampling plan to Nemotron packing blend JSON"
    )
    parser.add_argument("--sampling", required=True, help="Path to sampling plan JSON")
    parser.add_argument("--data-root", required=True,
                        help="Root directory containing category folders with JSONL files")
    parser.add_argument("--output", required=True, help="Output blend JSON path")
    args = parser.parse_args()

    sampling_to_blend(args.sampling, args.data_root, args.output)
