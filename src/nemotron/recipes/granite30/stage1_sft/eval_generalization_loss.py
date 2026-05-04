#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "granite30/eval_generalization_loss"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
# setup = "NeMo and all training dependencies are pre-installed in the image."
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.config]
# dir = "./config"
# format = "omegaconf"
# ///

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute generalization loss on held-out reasoning data for a single checkpoint.

The shell wrapper (run_eval_generalization.sh) loops over checkpoints and launches
a separate torchrun per checkpoint (Megatron distributed init is one-shot).

Usage:
    torchrun --nproc_per_node=4 eval_generalization_loss.py \
        --config config/eval_granite_8b_generalization.yaml \
        --checkpoint-load /path/to/checkpoints \
        --held-out-data /path/to/reasoning_holdout/splits \
        --step 500

    # With sampling (faster, uses fixed subset across checkpoints):
    torchrun --nproc_per_node=4 eval_generalization_loss.py \
        --config config/eval_granite_8b_generalization.yaml \
        --checkpoint-load /path/to/checkpoints \
        --held-out-data /path/to/reasoning_holdout/splits \
        --step 500 \
        --sample 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.training.config import ConfigContainer, FinetuningDatasetConfig
from megatron.bridge.training.eval import evaluate
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.setup import setup
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from omegaconf import OmegaConf

from nemotron.kit.recipe_loader import extract_recipe_config, import_recipe_function
from nemotron.kit.train_script import load_omegaconf_yaml

logger: logging.Logger = logging.getLogger(__name__)


def _print_rank_0(msg: str):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(msg, flush=True)


def sample_parquet_data(data_path: str, n_samples: int, seed: int = 42) -> str:
    """Sample N rows from parquet files and write to a temp directory.

    Uses a fixed seed so the same subset is evaluated across all checkpoints.

    Args:
        data_path: Directory containing .parquet files
        n_samples: Number of rows to sample
        seed: Random seed for reproducibility

    Returns:
        Path to temp directory containing sampled parquet file
    """
    import pyarrow.parquet as pq
    import pyarrow as pa
    import numpy as np

    data_dir = Path(data_path)
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_path}")

    _print_rank_0(f"Sampling {n_samples} rows from {len(parquet_files)} parquet file(s)...")

    tables = []
    for f in parquet_files:
        tables.append(pq.read_table(f))
    full_table = pa.concat_tables(tables)

    total_rows = len(full_table)
    actual_samples = min(n_samples, total_rows)
    _print_rank_0(f"  Total rows: {total_rows}, sampling: {actual_samples}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(total_rows, size=actual_samples, replace=False)
    indices.sort()
    sampled_table = full_table.take(indices)

    tmp_dir = tempfile.mkdtemp(prefix="eval_sample_")
    out_path = Path(tmp_dir) / "sampled.parquet"
    pq.write_table(sampled_table, out_path)
    _print_rank_0(f"  Written to: {tmp_dir}")

    return tmp_dir


def _resolve_val_path(held_out_data: str) -> str:
    """Resolve the actual directory containing parquet files."""
    held_out_path = Path(held_out_data)
    if (held_out_path / "valid").is_dir():
        return str(held_out_path / "valid")
    elif (held_out_path / "splits" / "valid").is_dir():
        return str(held_out_path / "splits" / "valid")
    return str(held_out_path)


def build_eval_config(
    base_config_path: str,
    checkpoint_load_path: str,
    val_data_path: str,
    seq_length: Optional[int] = None,
    eval_iters: Optional[int] = None,
    cli_overrides: Optional[list[str]] = None,
) -> ConfigContainer:
    """Build a ConfigContainer for eval-only mode on a single checkpoint."""
    config = load_omegaconf_yaml(base_config_path)

    recipe_target, recipe_kwargs = extract_recipe_config(
        config,
        default_target="megatron.bridge.recipes.granite.granite_8b.granite_8b_finetune_config",
    )
    recipe_func = import_recipe_function(recipe_target)
    cfg: ConfigContainer = recipe_func(**recipe_kwargs)

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    config_overrides = OmegaConf.to_container(config, resolve=False)
    config_overrides.pop("recipe", None)
    config_overrides.pop("run", None)
    config_overrides.pop("dataset", None)

    if config_overrides:
        yaml_overrides_omega = OmegaConf.create(config_overrides)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)

    if cli_overrides:
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    final_overrides_as_dict.pop("dataset", None)
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # -- Eval-only mode --
    cfg.validation.skip_train = True
    cfg.validation.eval_interval = None
    cfg.train.train_iters = 0

    if eval_iters is not None:
        cfg.validation.eval_iters = eval_iters

    # -- Checkpoint --
    cfg.checkpoint.load = checkpoint_load_path
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.finetune = False
    cfg.checkpoint.save = None
    cfg.checkpoint.save_interval = None

    # -- Dataset --
    effective_seq_length = seq_length or cfg.model.seq_length

    packed_specs = PackedSequenceSpecs(
        packed_sequence_size=effective_seq_length,
        packed_train_data_path=None,
        packed_val_data_path=val_data_path,
        packed_metadata_path=None,
    )

    if cfg.model.context_parallel_size > 1:
        packed_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    cfg.dataset = FinetuningDatasetConfig(
        dataset_root=None,
        seq_length=effective_seq_length,
        packed_sequence_specs=packed_specs,
        dataloader_type="batch",
        do_validation=True,
        do_test=False,
    )

    # -- Disable logging --
    if hasattr(cfg, "logger"):
        cfg.logger.wandb_project = None
        cfg.logger.wandb_exp_name = None

    cfg.ddp.use_distributed_optimizer = False

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Compute generalization loss for a single checkpoint on held-out reasoning data"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-load", type=str, required=True,
                        help="Checkpoint directory (parent of iter_XXXXXXX/)")
    parser.add_argument("--held-out-data", type=str, required=True,
                        help="Held-out validation data (packed parquet)")
    parser.add_argument("--step", type=int, required=True,
                        help="Training step of this checkpoint (for logging)")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON file to append results to")
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None,
                        help="Number of eval iterations (default: config value)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N rows from validation data")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Seed for sampling (same subset across all checkpoints)")

    args, unknown_args = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)

    _print_rank_0(f"\n{'='*60}")
    _print_rank_0(f"Evaluating generalization loss — step {args.step}")
    _print_rank_0(f"Checkpoint: {args.checkpoint_load}")
    _print_rank_0(f"Held-out data: {args.held_out_data}")
    if args.sample:
        _print_rank_0(f"Sampling: {args.sample} rows (seed={args.sample_seed})")
    _print_rank_0(f"{'='*60}")

    # Resolve validation data path
    resolved_val_path = _resolve_val_path(args.held_out_data)

    # Sample if requested (rank 0 samples, broadcasts path to all ranks)
    sampled_tmp_dir = None
    if args.sample:
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            sampled_tmp_dir = sample_parquet_data(resolved_val_path, args.sample, seed=args.sample_seed)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                path_bytes = sampled_tmp_dir.encode("utf-8")
                path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8).cuda()
                length_tensor = torch.tensor([len(path_bytes)], dtype=torch.long).cuda()
            else:
                length_tensor = torch.zeros(1, dtype=torch.long).cuda()
                path_tensor = None
            torch.distributed.broadcast(length_tensor, src=0)
            if torch.distributed.get_rank() != 0:
                path_tensor = torch.zeros(length_tensor.item(), dtype=torch.uint8).cuda()
            torch.distributed.broadcast(path_tensor, src=0)
            sampled_tmp_dir = bytes(path_tensor.cpu().tolist()).decode("utf-8")

        eval_data_path = sampled_tmp_dir
    else:
        eval_data_path = resolved_val_path

    # Build config and run eval
    cfg = build_eval_config(
        base_config_path=args.config,
        checkpoint_load_path=args.checkpoint_load,
        val_data_path=eval_data_path,
        seq_length=args.seq_length,
        eval_iters=args.eval_iters,
        cli_overrides=unknown_args if unknown_args else None,
    )

    state = GlobalState(cfg)

    from megatron.bridge.training.data_provider import get_dataset_provider

    dataset_provider = get_dataset_provider(cfg.dataset)
    setup_output = setup(state, dataset_provider)

    state = setup_output.state
    model = setup_output.model
    valid_data_iterator = setup_output.valid_data_iterator

    if valid_data_iterator is None:
        _print_rank_0("ERROR: No validation data iterator created.")
        sys.exit(1)

    total_loss_dict, _, timelimit = evaluate(
        state=state,
        forward_step_func=forward_step,
        data_iterator=valid_data_iterator,
        model=model,
        process_non_loss_data_func=None,
        config=cfg,
        verbose=True,
    )

    if timelimit or total_loss_dict is None:
        _print_rank_0("ERROR: Evaluation hit timelimit or returned no results")
        sys.exit(1)

    # Report results
    results = {"step": args.step}
    for key, value in total_loss_dict.items():
        loss_val = value.item()
        ppl = math.exp(min(20, loss_val))
        results[key.replace(" ", "_")] = loss_val
        results[f"{key.replace(' ', '_')}_ppl"] = ppl
        _print_rank_0(f"  {key}: {loss_val:.6f} (PPL: {ppl:.2f})")

    # Write to JSON
    if args.output and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing = {}
        if output_path.exists():
            existing = json.loads(output_path.read_text())
        existing[str(args.step)] = results
        output_path.write_text(json.dumps(existing, indent=2))
        _print_rank_0(f"Results appended to {args.output}")

    _print_rank_0(f"RESULT step={args.step} loss={results.get('lm_loss', 0.0):.6f}")

    # Cleanup
    if sampled_tmp_dir and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        shutil.rmtree(sampled_tmp_dir, ignore_errors=True)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
