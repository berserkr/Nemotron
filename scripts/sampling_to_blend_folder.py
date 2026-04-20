#!/usr/bin/env python3
"""Convert a sampling plan JSON to a folder-level Nemotron blend JSON.

One blend entry per category folder. Weight = target / 1000.

Usage:
    python sampling_to_blend_folder.py \
        --sampling /path/to/sampling_plan.json \
        --data-root /proj/datasets/sft_datasets/granite-4.2-sft-datasets-r260414a/v1 \
        --output /path/to/blend.json
"""

import argparse
import json
import sys
from pathlib import Path


def find_category_dir(data_root: Path, category: str) -> Path | None:
    """Find the category directory, trying common name variations."""
    for variant in [category, category.replace("_", "-"), category.replace("-", "_")]:
        candidate = data_root / variant
        if candidate.is_dir():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert sampling plan to folder-level blend JSON")
    parser.add_argument("--sampling", required=True, help="Path to sampling plan JSON")
    parser.add_argument("--data-root", required=True, help="Root directory containing category folders")
    parser.add_argument("--output", required=True, help="Output blend JSON path")
    args = parser.parse_args()

    with open(args.sampling) as f:
        plan = json.load(f)

    root = Path(args.data_root)
    version = plan.get("version", "unknown")
    total_target = plan.get("total_samples", 0)
    sampling = plan.get("sampling", {})

    datasets = []
    found = 0
    missing = []

    for category, info in sorted(sampling.items()):
        target = info.get("target", 0)
        strategy = info.get("strategy", "unknown")
        ratio = info.get("ratio", 1.0)
        meta_category = info.get("category", "unknown")
        available = info.get("available", "?")

        cat_dir = find_category_dir(root, category)
        if cat_dir is None:
            missing.append(category)
            print(f"  MISSING: {category}", file=sys.stderr)
            continue

        # Count files and samples
        jsonl_files = sorted(cat_dir.glob("*.jsonl"))
        num_files = len(jsonl_files)
        total_size_gb = sum(f.stat().st_size for f in jsonl_files) / (1 << 30)

        found += 1
        weight = round(target / 1000.0, 1)

        datasets.append({
            "_category": meta_category,
            "_strategy": f"{strategy} ({ratio}x)",
            "_target": target,
            "_available": available,
            "_files": num_files,
            "_size_gb": round(total_size_gb, 1),
            "name": category,
            "path": f"file://{cat_dir}",
            "weight": weight,
        })

        print(f"  {category:<45s} target={target:>10,}  weight={weight:>8.1f}  files={num_files:>3}  size={total_size_gb:.1f}GB")

    blend = {
        "_comment": f"Folder-level blend from sampling plan: {version}. Target: {total_target:,} samples.",
        "_version": version,
        "_total_target_samples": total_target,
        "_category_distribution": plan.get("category_distribution", {}),
        "datasets": datasets,
    }

    if missing:
        blend["_missing_categories"] = missing

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(blend, f, indent=2)

    print(f"\nCategories: {found} found, {len(missing)} missing")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    total_weight = sum(d["weight"] for d in datasets)
    print(f"Total weight: {total_weight:.1f}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
