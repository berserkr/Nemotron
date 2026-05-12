#!/usr/bin/env python3
"""Repair split symlinks for packed SFT data.

Problem: dataset-prefixed symlink names (e.g. code_2__shard_000029.parquet)
cause data to be read grouped by dataset when the loader sorts alphabetically,
defeating the shuffle that blend.json provides.

This script:
  1. Reads blend.json from the output directory (already in shuffled order)
  2. Clears existing symlinks from splits/{train,valid,test}/
  3. Recreates symlinks with sequential numbering (shard_000000.parquet, ...)
     preserving the shuffled order from blend.json so cross-dataset mixing
     is maintained regardless of how the data loader orders files.

Usage:
    python fix_split_symlinks.py /path/to/output_dir
    python fix_split_symlinks.py /path/to/output_dir --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def fix_split_symlinks(output_dir: Path, dry_run: bool = False) -> dict[str, dict[str, int]]:
    """Regenerate split symlinks with sequential naming preserving shuffle order.

    Args:
        output_dir: Root output directory containing blend.json and splits/
        dry_run: If True, print what would be done without making changes.

    Returns:
        Per-split stats: {"train": {"created": N, "missing": M}, ...}
    """
    blend_path = output_dir / "blend.json"
    if not blend_path.exists():
        print(f"ERROR: blend.json not found at {blend_path}", file=sys.stderr)
        sys.exit(1)

    with open(blend_path) as f:
        blend_data = json.load(f)

    splits_base = output_dir / "splits"
    results: dict[str, dict[str, int]] = {}

    for split_name, path_list in blend_data.items():
        split_dir = splits_base / split_name
        shard_paths = [path_list[i] for i in range(1, len(path_list), 2)]

        if not shard_paths:
            results[split_name] = {"created": 0, "missing": 0, "removed": 0}
            continue

        # Phase 1: Remove all existing symlinks in the split dir
        removed = 0
        if split_dir.exists() and not dry_run:
            for entry in split_dir.iterdir():
                if entry.is_symlink() or entry.is_file():
                    entry.unlink()
                    removed += 1
        elif split_dir.exists():
            removed = sum(1 for e in split_dir.iterdir() if e.is_symlink() or e.is_file())

        if not dry_run:
            split_dir.mkdir(parents=True, exist_ok=True)

        # Phase 2: Create symlinks with sequential index (preserves blend.json shuffle)
        created = 0
        missing = 0

        for seq_idx, shard_path_str in enumerate(shard_paths):
            parquet_path_str = f"{shard_path_str}.parquet"
            parquet_path = Path(parquet_path_str)

            if not parquet_path.exists():
                missing += 1
                print(f"  MISSING: {parquet_path_str}")
                continue

            # Sequential naming preserves the shuffled order from blend.json
            link_name = f"shard_{seq_idx:06d}.parquet"
            link_path = split_dir / link_name

            if dry_run:
                print(f"  WOULD CREATE: {link_path} -> .../{parquet_path.parent.parent.name}/.../{parquet_path.name}")
            else:
                try:
                    rel_target = os.path.relpath(parquet_path, split_dir)
                    link_path.symlink_to(rel_target)
                except OSError:
                    link_path.symlink_to(parquet_path.resolve())
            created += 1

        results[split_name] = {"created": created, "missing": missing, "removed": removed}
        action = "Would create" if dry_run else "Created"
        print(f"  {split_name}: {action} {created} symlinks, removed {removed} old, {missing} missing files")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fix split symlinks for packed SFT data")
    parser.add_argument("output_dir", type=Path, help="Root output directory (contains blend.json)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    print(f"{'DRY RUN - ' if args.dry_run else ''}Fixing split symlinks in: {output_dir}")
    print()

    results = fix_split_symlinks(output_dir, dry_run=args.dry_run)

    print()
    print("Summary:")
    total_created = sum(r["created"] for r in results.values())
    total_missing = sum(r["missing"] for r in results.values())
    print(f"  Total symlinks created: {total_created}")
    print(f"  Total missing parquet files: {total_missing}")

    if total_missing > 0:
        print(f"\n  WARNING: {total_missing} shard files were not found on disk.", file=sys.stderr)
        print("  The pipeline may not have completed for those shards.", file=sys.stderr)


if __name__ == "__main__":
    main()
