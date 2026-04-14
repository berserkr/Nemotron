# Granite Training Guide

End-to-end guide for setting up repositories, packing data, training with
context parallelism, and exporting checkpoints back to HuggingFace format.

Supports Granite 3B Dense, 8B Dense, MoE 3B, and 30B Dense models.

**Cluster**: GB200 nodes, 4 GPUs per node (184 GB each)
**Container**: `/mnt/vast/squash/nemo_sft_python312_v4.sqsh`

---

## Model Overview

| Model | Arch | Params | Layers | Hidden | FFN | Heads | KV Heads | Experts | Vocab | rope_theta | Max Ctx |
|-------|------|--------|--------|--------|-----|-------|----------|---------|-------|------------|---------|
| Granite 3B Dense | GraniteForCausalLM | ~3B | 40 | 2560 | 8192 | 40 | 8 | - | 100352 | 10M | 128k |
| Granite 8B Dense | GraniteForCausalLM | ~8B | 32 | 4096 | 14336 | 32 | 8 | - | 49152* | 10M | 128k |
| Granite MoE 3B | GraniteMoeForCausalLM | ~3B (800M active) | 32 | 1536 | 512/expert | 24 | 8 | 40 (top-8) | 49152* | 10k** | 4k** |
| Granite 30B Dense | GraniteForCausalLM | ~28.8B | 64 | 4096 | 32768 | 32 | 8 | - | 100352 | 50M | 512k |

\* Granite 8B (3.1/3.3) has vocab=49159 (odd) — incompatible with TP=2. Use TP=1.
\*\* MoE 3B base model needs `rotary_base: 500000` override for >40k context. Instruct model has rope_theta=10M.

### Granite Multipliers (muP)

All Granite models use muP-style scaling multipliers stored in the HF config. The bridge bakes these into weights during loading:

| Multiplier | Dense 3B | Dense 8B | MoE 3B | Dense 30B | Applied to |
|-----------|----------|----------|--------|-----------|------------|
| embedding_multiplier | 12.0 | varies | 12.0 | varies | embed_tokens × multiplier |
| residual_multiplier | 0.22 | varies | 0.22 | varies | o_proj, down_proj × multiplier |
| logits_scaling | 10.0 | varies | 6.0 | varies | lm_head / scaling |
| attention_multiplier | 0.015625 | varies | 0.015625 | varies | softmax_scale (NOT baked) |

---

## Validated Training Configurations

| Model | TP | PP | EP | CP | Seq Len | Nodes | GPUs | Status |
|-------|----|----|----|----|---------|-------|------|--------|
| Granite 3B Dense | 1 | 1 | - | 1 | 128k | 1 | 4 | TP=1, DP=4 |
| Granite 3B Dense | 1 | 1 | - | 2 | 256k | 1 | 4 | TP=1, CP=2, DP=2 |
| Granite 8B Dense | 1 | 1 | - | 1 | 8k | 1 | 4 | TP=1, DP=4 |
| Granite 8B Dense | 4 | 1 | - | 2 | 128k | 4 | 16 | TP=4, CP=2, DP=2 |
| Granite 8B Dense | 2 | 1 | - | 4 | 256k | 4 | 16 | TP=2, CP=4, DP=2 |
| Granite 8B Dense | 4 | 8 | - | 16 | 256k | 128 | 512 | stable |
| Granite 8B Dense | 4 | 4 | - | 32 | 512k | 128 | 512 | stable |
| MoE 3B instruct | 1 | 1 | 4 | 1 | 8k | 1 | 4 | loss 1.37→0.65 |
| MoE 3B base | 1 | 1 | 8 | 4 | 128k | 8 | 32 | needs rotary_base=500000 |
| Granite 30B Dense | 4 | 8 | - | 16 | 256k | 128 | 512 | stable |

### Known Failures

| Model | Config | Issue |
|-------|--------|-------|
| MoE 3B | TP=2 + vocab resize | loss=26 (bridge bug with non-original vocab + TP>1) |
| MoE 3B | EP=2 + CP=2 | NaN step 1 (MoE + small EP + CP interaction) |
| Granite 8B (3.1/3.3) | TP=2 | vocab=49159 (odd) — not divisible by TP=2 |

### Key Lessons

- **MoE + CP**: only works with larger EP (8+), fails with EP=2
- **MoE + TP>1**: broken with non-original vocab size — use TP=1 or keep original vocab
- **MoE base model >40k ctx**: needs `rotary_base: 500000` override (original 10k too low)
- **Granite 8B (3.1/3.3)**: vocab=49159 (odd) — use TP=1 only
- **Granite 3B Dense**: vocab=100352 — divisible by common TP sizes, no issues
- **NVLink errors on GB200**: hardware issue, use pre-flight checks and `--exclude` bad nodes

---

## Model Paths, Configs & Launch Scripts

### Base Models

| Model | Path |
|-------|------|
| Granite 3B Dense (instruct) | `/mnt/vast/proj/checkpoints/bathen/models/base/granite-3.3-3b-instruct` |
| Granite 8B Dense (instruct) | `/mnt/vast/proj/checkpoints/bathen/models/base/granite-3.3-8b-instruct` |
| Granite 8B Dense (base) | `/mnt/vast/proj/checkpoints/bathen/models/base/granite-3.3-8b-base` |
| Granite MoE 3B (instruct) | `/mnt/vast/proj/checkpoints/bathen/models/base/granite-3.0-3b-a800m-instruct` |
| Granite MoE 3B (base) | `/mnt/vast/proj/checkpoints/bathen/models/base/granite-3.0-3b-a800m-base` |
| Granite 30B Dense | `/mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7` |

### Training Configs

Located in `src/nemotron/recipes/granite30/stage1_sft/config/`:

| Config File | Model | Seq Len | Nodes |
|------------|-------|---------|-------|
| `train_granite_3b_128k.yaml` | 3B Dense | 128k | 1 |
| `train_granite_3b_256k.yaml` | 3B Dense | 256k | 1 |
| `train_granite_8b_4k.yaml` | 8B Dense | 4k | 16 |
| `train_granite_8b_128k.yaml` | 8B Dense | 128k | 16 |
| `train_granite_8b_256k.yaml` | 8B Dense | 256k | 4 |
| `train_granite_moe_3b_64k.yaml` | MoE 3B | 64k | 8 |
| `train_granite_moe_3b_128k.yaml` | MoE 3B | 128k | 8 |
| `train_granite_30b_128k.yaml` | 30B Dense | 128k | 64 |
| `train_granite_30b_256k.yaml` | 30B Dense | 256k | 128 |
| `train_granite_30b_512k.yaml` | 30B Dense | 512k | 256 |

### Launch Scripts

Located in the Nemotron root directory:

| Script | Model | Seq Len | Nodes |
|--------|-------|---------|-------|
| `launch_granite_3b_128k.sh` | 3B Dense | 128k | 1 |
| `launch_granite_8b_4k.sh` | 8B Dense | 4k | 16 |
| `launch_granite_8b_128k.sh` | 8B Dense | 128k | 16 |
| `launch_granite_moe_3b_64k.sh` | MoE 3B | 64k | 8 |
| `launch_granite_moe_3b_128k.sh` | MoE 3B | 128k | 8 |
| `launch_granite_30b_128k.sh` | 30B Dense | 128k | 64 |
| `launch_granite_30b_256k.sh` | 30B Dense | 256k | 128 |
| `launch_granite_30b_512k.sh` | 30B Dense | 512k | 256 |

### Bridge Recipes

Located in `Megatron-Bridge/src/megatron/bridge/recipes/granite/`:

| Recipe | Model |
|--------|-------|
| `granite_3b.py` | Granite 3B Dense (`granite_3b_finetune_config`) |
| `granite_8b.py` | Granite 8B Dense (`granite_8b_finetune_config`) |
| `granite_moe_3b.py` | Granite MoE 3B (`granite_moe_3b_finetune_config`) |
| `granite_30b.py` | Granite 30B Dense (`granite_30b_finetune_config`) |

### Chat Templates

Located in the Nemotron root directory:

| Template | Format |
|----------|--------|
| `chat_template.jinja` | `<|im_start|>` / `<|im_end|>` format |
| `chat_template_granite_instruct.jinja` | `<|start_of_role|>` / `<|end_of_role|>` / `<|end_of_text|>` format |
| `chat_template_granite_full.jinja` | Full Granite instruct with tools, documents, citations, thinking |

---

## 0. Repository Setup

### Clone and Checkout

Both repositories must be on the `granite_v1` branch:

```bash
# Choose your workspace (adjust to your home directory)
export WORKDIR=/mnt/home/$USER/src/github.com
mkdir -p $WORKDIR && cd $WORKDIR

# Clone Megatron-Bridge
git clone git@github.com:berserkr/Megatron-Bridge.git
cd Megatron-Bridge
git checkout granite_v1
git submodule update --init --recursive   # pulls Megatron-LM under 3rdparty/
cd Megatron-Bridge/3rdparty

# Make sure 3rdparty points to the right Megatron-LM branch (delete if already there)
git clone --recursive git@github.com:berserkr/Megatron-LM.git
git checkout super_cp2_fixes
cd ../..

# Clone Nemotron
git clone git@github.com:berserkr/Nemotron.git
cd Nemotron
git checkout granite_v1
cd ..

```

### Set PYTHONPATH

The container needs to find both repositories. Add to your launch script
(before `srun`), or to your `run.env`:

```bash
export PYTHONPATH=/mnt/home/$USER/src/github.com/Megatron-Bridge/src:\
/mnt/home/$USER/src/github.com/Megatron-Bridge/3rdparty/Megatron-LM:\
/mnt/home/$USER/src/github.com/Nemotron/src:\
${PYTHONPATH}
```

### Update Paths in Scripts

All scripts and configs in this guide use placeholder paths. You **must**
update the following to match your environment:

| Placeholder | What to change |
|------------|----------------|
| `/mnt/home/$USER/src/github.com/Megatron-Bridge` | Your Megatron-Bridge clone path |
| `/mnt/home/$USER/src/github.com/Nemotron` | Your Nemotron clone path |
| `/mnt/vast/proj/checkpoints/$USER/...` | Your checkpoint/data storage paths |
| `/mnt/vast/squash/nemo_sft_0331.sqsh` | Container image (shared, no change needed) |

In SLURM scripts, update these specifically:

```bash
# Container image (use this exact path)
container_image="/mnt/vast/squash/nemo_sft_0331.sqsh"

# Working directory inside the container — point to YOUR Nemotron clone
--container-workdir=/mnt/home/$USER/src/github.com/Nemotron

# For conversion scripts, workdir should be YOUR Megatron-Bridge clone
--container-workdir=/mnt/home/$USER/src/github.com/Megatron-Bridge
```

In training configs, update:

```yaml
recipe:
    hf_model_path: /mnt/vast/proj/checkpoints/$USER/models/base/<your-base-model>

dataset:
    super3_packed_sft_dir: /mnt/vast/proj/checkpoints/$USER/datasets/sft/<your-packed-data>/splits

checkpoint:
    save: /mnt/vast/proj/checkpoints/$USER/models/nemo_run/<your-experiment>
    load: /mnt/vast/proj/checkpoints/$USER/models/nemo_run/<your-experiment>
```

### Verify Setup

Quick sanity check inside the container:

```bash
srun --nodes=1 --ntasks=1 --gpus-per-node=1 \
    --container-image=/mnt/vast/squash/nemo_sft_0331.sqsh \
    --container-mounts=/mnt:/mnt \
    --container-workdir=/mnt/home/$USER/src/github.com/Nemotron \
    bash -c "python -c 'import megatron.bridge; print(\"Megatron-Bridge OK\"); \
             from nemotron.kit import print_step_complete; print(\"Nemotron OK\")'"
```

---

## 1. Data Packing

### The pad_seq_to_mult Formula

When using packed sequences with Context Parallelism (CP) and Sequence
Parallelism (SP), sub-sequences within each pack must be padded so that
the total tokens per CP rank is divisible by the Tensor Parallel (TP) size.

**Correct formula:**

```
pad_seq_to_mult = TP x CP
```

> The recipe default `CP x 2` is **wrong** and will cause assertion errors
> at higher CP values. Always use `TP x CP`.

### Reference Table (TP=4)

| Target CP | TP x CP | pad_seq_to_mult |
|-----------|---------|-----------------|
| 1         | 4       | 4               |
| 2         | 8       | 8               |
| 4         | 16      | 16              |
| 8         | 32      | 32              |
| 16        | 64      | 64              |
| 32        | 128     | 128             |
| 64        | 256     | 256             |

### Future-proofing

Pack with `pad_seq_to_mult=256` to support any CP up to 64 without
repacking. The extra padding per sub-sequence is negligible (~6% waste at
4k average sub-sequence length).

A `pad_seq_to_mult=N` dataset is compatible with **any** TP x CP that
divides N. For example, data packed with `pad_seq_to_mult=256` works for
CP=8 (TP x CP=32), CP=16 (64), CP=32 (128), and CP=64 (256).

### Packing Config Example

Config YAML for the data prep script
(`Nemotron/src/nemotron/recipes/super3/stage1_sft/data_prep.py`):

```yaml
# --- 256k context, future-proof for CP up to 64 ---
blend_path: /mnt/vast/proj/checkpoints/bathen/datasets/sft/datasets_config_128k.json
output_dir: /mnt/vast/proj/checkpoints/bathen/datasets/sft/granite_30b_tok_256k_pad256

num_shards: 128

tokenizer:
  model: /mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7
  add_bos: false
  add_eos: true

pack_size: 262144          # 256k tokens per pack
algorithm: first_fit_shuffle
pad_seq_to_mult: 256       # TP(4) x CP(64) -- covers all CP up to 64

seed: null
parquet_row_group_size: 1000
parquet_compression: zstd

train_ratio: 0.98
valid_ratio: 0.01
test_ratio: 0.01

chat_template: /mnt/home/bathen/src/github.com/Nemotron/chat_template.jinja
messages_field: messages
tools_field: tools
```

For 512k context, change only:

```yaml
pack_size: 524288          # 512k
output_dir: /mnt/vast/proj/checkpoints/bathen/datasets/sft/granite_30b_tok_512k_pad256
# pad_seq_to_mult stays 256
```

### What pad_seq_to_mult Does

The packing script pads each **individual sub-sequence** to a multiple of
this value before bin-packing them into the `pack_size`. It does NOT know
about TP or CP -- it is a plain integer. The constraint comes from training:

1. CP splits the packed sequence across CP ranks at sub-sequence boundaries
2. Each CP rank gets a sum of sub-sequence lengths
3. SP reduce-scatters that sum across TP ranks
4. That sum must be divisible by TP

Since each sub-sequence is a multiple of `pad_seq_to_mult`, any sum of
sub-sequences is also a multiple of `pad_seq_to_mult`. For the sum to be
divisible by TP, `pad_seq_to_mult` must be divisible by TP. In practice,
`TP x CP` guarantees correctness for all possible partitionings.

---

## 2. Training

### Parallelism Planning

With GB200 nodes (4 GPUs each):

```
Total GPUs = nodes x 4
Total GPUs = TP x PP x CP x DP
```

Target **16k tokens per GPU** for Granite 30B to avoid OOM:

```
tokens_per_gpu = seq_length / CP
```

### Configs by Sequence Length

#### 128k context -- 64 nodes (256 GPUs)

TP=4, PP=16, CP=4 = 256 GPUs, DP=1

```yaml
recipe:
    _target_: megatron.bridge.recipes.granite.granite_30b.granite_30b_finetune_config
    hf_model_path: /mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 16
    context_parallelism: 4
    sequence_parallelism: true
    seq_length: 131072
    micro_batch_size: 1
    global_batch_size: 64
    finetune_lr: 5.0e-6
    lr_warmup_iters: 50
    train_iters: 2000
    packed_sequence: true

  # Do NOT override recompute -- recipe uses selective for CP>1

ddp:
    use_distributed_optimizer: true

dataset:
    seq_length: 131072
    super3_packed_sft_dir: /mnt/vast/proj/checkpoints/bathen/datasets/sft/granite_30b_tok_cp4/splits
    packed_sequence_specs:
        packed_sequence_size: 131072
        pad_seq_to_mult: 16    # TP(4) x CP(4) - must match padding in packing step

train:
    train_iters: 2000
    global_batch_size: 64

checkpoint:
    save: /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite30b_sft_128k
    load: /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite30b_sft_128k
    save_interval: 250
    finetune: true
    most_recent_k: 3

logger:
    log_interval: 1
    wandb_project: "granite30b_sft_128k"
    wandb_exp_name: "granite30b_sft_128k_cp4"
    wandb_entity: "bathen"
```

#### 256k context -- 128 nodes (512 GPUs)

TP=4, PP=8, CP=16 = 512 GPUs, DP=1

```yaml
recipe:
    _target_: megatron.bridge.recipes.granite.granite_30b.granite_30b_finetune_config
    hf_model_path: /mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 8
    context_parallelism: 16
    sequence_parallelism: true
    seq_length: 262144
    micro_batch_size: 1
    global_batch_size: 64
    finetune_lr: 5.0e-6
    lr_warmup_iters: 50
    train_iters: 2000
    packed_sequence: true

  # Do NOT override recompute -- recipe uses selective for CP>1

ddp:
    use_distributed_optimizer: true

dataset:
    seq_length: 262144
    super3_packed_sft_dir: /mnt/vast/proj/checkpoints/bathen/datasets/sft/granite_30b_tok_256k_pad256/splits
    packed_sequence_specs:
        packed_sequence_size: 262144
        pad_seq_to_mult: 256   # future-proof

train:
    train_iters: 2000
    global_batch_size: 64

checkpoint:
    save: /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite30b_sft_256k
    load: /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite30b_sft_256k
    save_interval: 250
    finetune: true
    most_recent_k: 3

logger:
    log_interval: 1
    wandb_project: "granite30b_sft_256k"
    wandb_exp_name: "granite30b_sft_256k_cp16"
    wandb_entity: "bathen"
```

#### 512k context -- 256 nodes (1024 GPUs)

TP=4, PP=8, CP=32 = 1024 GPUs, DP=1

```yaml
recipe:
    _target_: megatron.bridge.recipes.granite.granite_30b.granite_30b_finetune_config
    hf_model_path: /mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 8
    context_parallelism: 32
    sequence_parallelism: true
    seq_length: 524288
    micro_batch_size: 1
    global_batch_size: 64
    finetune_lr: 5.0e-6
    lr_warmup_iters: 50
    train_iters: 2000
    packed_sequence: true

  # Do NOT override recompute -- recipe uses selective for CP>1

ddp:
    use_distributed_optimizer: true

dataset:
    seq_length: 524288
    super3_packed_sft_dir: /mnt/vast/proj/checkpoints/bathen/datasets/sft/granite_30b_tok_512k_pad256/splits
    packed_sequence_specs:
        packed_sequence_size: 524288
        pad_seq_to_mult: 256

train:
    train_iters: 2000
    global_batch_size: 64

checkpoint:
    save: /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite30b_sft_512k
    load: /mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite30b_sft_512k
    save_interval: 250
    finetune: true
    most_recent_k: 3

logger:
    log_interval: 1
    wandb_project: "granite30b_sft_512k"
    wandb_exp_name: "granite30b_sft_512k_cp32"
    wandb_entity: "bathen"
```

All training scripts that worked so far are here:

```bash
-rw-rw-r-- 1 bathen bathen 5072 Mar 22 05:29 launch_qwen3_8b_256k_cp.sh
-rw-rw---- 1 bathen bathen 5131 Mar 22 23:46 launch_super3_256k_cp.sh
-rw-rw-r-- 1 bathen bathen 5072 Mar 25 21:50 launch_qwen3_8b_128k_cp.sh
-rw-rw-r-- 1 bathen bathen 5069 Mar 27 04:04 launch_qwen3_8b_128k.sh
-rw-rw---- 1 bathen bathen 5166 Mar 27 20:39 launch_granite_8b_4k.sh
-rw-rw---- 1 bathen bathen 5168 Mar 27 21:02 launch_granite_8b_128k.sh
-rw-rw---- 1 bathen bathen 5205 Mar 28 05:33 launch_granite_30b_128k.sh
-rw-rw-r-- 1 bathen bathen 5389 Mar 31 15:43 launch_granite_30b_256k.sh
-rw-rw-r-- 1 bathen bathen 5389 Mar 31 15:58 launch_granite_30b_512k.sh
```

All contain the individual configs used.

Curves (30b bridge): 
```
https://wandb.vpc.res.ibm.com/bathen/granite30b_sft_128k/runs/scgcz8tr?nw=nwuserbathen

```

CP1 vs CP2:
```
https://wandb.vpc.res.ibm.com/bathen/full_sft_qwen3_8b_128k_validation/runs/qi9hmrgh?nw=nwuserbathen
https://wandb.vpc.res.ibm.com/bathen/full_sft_qwen3_8b_128k_validation/runs/3jgfcr1t?nw=nwuserbathen
```

### Important Training Notes

**Do NOT override recompute settings.** The recipe (`granite_30b.py`)
automatically uses selective recompute when CP > 1. Overriding with
`recompute_granularity: "full"` / `recompute_num_layers: 1` stores
activations for most layers and causes OOM.

**NFS file lock fix.** Add to your launch script before `srun`:

```bash
export MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron_locks
export FILELOCK_LOCK_CLASS=soft
```

The default lock directory (`~/.cache/huggingface/`) is on NFS/VAST and
`fcntl.flock()` fails at scale (128+ nodes). This puts locks on local
`/tmp` instead.

**NCCL environment variables** for GB200 (in `run.env`):

```bash
export NCCL_TIMEOUT=300000                    # 5 min (ms), NCCL-level
export NCCL_TIMEOUT_WAIT_SEC=300              # 5 min
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300   # 5 min
export TORCH_NCCL_ENABLE_TIMING=1
export TORCH_NCCL_DESYNC_DEBUG=1              # logs which rank is desynced
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

**Symlink fix for requeued jobs:**

```bash
ln -sfn /tmp/$USER/triton ~/.triton    # -sfn prevents failure if exists
```

### Quick Reference: Parallelism Planner

| Seq Length | Nodes | GPUs | TP | PP | CP  | DP | Tokens/GPU |
|------------|-------|------|----|----|-----|----|------------|
| 128k       | 64    | 256  | 4  | 16 | 4   | 1  | 32k        |
| 256k       | 128   | 512  | 4  | 8  | 16  | 1  | 16k        |
| 512k       | 256   | 1024 | 4  | 8  | 32  | 1  | 16k        |

---

## 3. Checkpoint Conversion (Megatron to HuggingFace)

### Export Script

The conversion requires `TP x PP` GPUs. Granite uses tied embeddings,
so `--not-strict` is always required.

| Training Config | TP | PP | GPUs Needed | Nodes (4 GPU/node) |
|----------------|----|----|-------------|---------------------|
| 128k           | 4  | 16 | 64          | 16                  |
| 256k           | 4  | 8  | 32          | 8                   |
| 512k           | 4  | 8  | 32          | 8                   |

### SLURM Export Script (256k example)

```bash
#!/bin/bash
#SBATCH --partition=hpc-mid
#SBATCH --nodes=8
#SBATCH --job-name=granite30b-export
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144
#SBATCH --output="/mnt/vast/proj/checkpoints/bathen/logs/export-out.%j.log"
#SBATCH --error="/mnt/vast/proj/checkpoints/bathen/logs/export-err.%j.log"
#SBATCH --wait-all-nodes=1
#SBATCH --mem=0

. ~/.bashrc
source ~/run.env

export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1
export NCCL_COLLNET_ENABLE=0
export NVIDIA_IMEX_CHANNELS=0
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"
export MASTER_PORT=28444
export NNODES=$SLURM_NNODES

# --- PATHS (update these) ---
LOCAL_HF_CKPT=/mnt/vast/proj/checkpoints/bathen/models/base/30b-soft-lc-512k-lr1e-4-merged-3-7
SAVED_CKPT=/mnt/vast/proj/checkpoints/bathen/models/nemo_run/granite30b_sft_256k/iter_0002000
EXPORTED_CKPT=/mnt/vast/proj/checkpoints/bathen/models/exports/granite30b_sft_256k/

container_mounts="/mnt:/mnt"
container_image="/mnt/vast/squash/nemo_sft_python312_v4.sqsh"

SRUN_ARGS="--kill-on-bad-exit=1 \
            --container-image=${container_image} \
            --container-mounts=${container_mounts} \
            --no-container-remap-root \
            --container-workdir=/mnt/home/bathen/src/github.com/Megatron-Bridge"

export DISTRIBUTED_ARGS=" \
    --nnodes ${NNODES} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank \$SLURM_NODEID \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}"

# --not-strict is REQUIRED for Granite (tied embeddings)
# --tp and --pp must match the TRAINING config
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun ${DISTRIBUTED_ARGS} \
    examples/conversion/convert_checkpoints_multi_gpu.py export \
    --hf-model ${LOCAL_HF_CKPT} \
    --megatron-path ${SAVED_CKPT} \
    --hf-path ${EXPORTED_CKPT} \
    --tp 4 --pp 8 \
    --not-strict"

echo "$(date) Starting export: ${SAVED_CKPT} -> ${EXPORTED_CKPT}"
srun ${SRUN_ARGS} bash -c "${CMD}"
echo "rc=$?"
```

### Adapting for Other Sequence Lengths

For 128k checkpoints (TP=4, PP=16): change `--nodes=16`, `--pp 16`,
and update `SAVED_CKPT` / `EXPORTED_CKPT` paths.

For 512k checkpoints (TP=4, PP=8): same as 256k (both use PP=8),
just update paths.

### Why --not-strict?

Granite has `tie_word_embeddings=True` in HuggingFace -- the embedding
and lm_head share the same weight tensor. AutoBridge **unties** them
during training (separate embed + output_layer with different Granite
scaling multipliers). During export, the converter generates a separate
`lm_head.weight` that doesn't exist in the original HF model structure.
`--not-strict` allows this mismatch. The bridge correctly un-bakes the
Granite multipliers during export.

---

## 4. Troubleshooting

### NFS File Lock Errors at Scale (128+ nodes)

**Symptom:**

```
OSError: [Errno 116] Stale file handle
OSError: [Errno 37] No locks available
ValueError: Failed to load configuration from ... after 4 attempts.
```

**Cause:** `safe_config_loader.py` uses `filelock.FileLock` which calls
`fcntl.flock()` — this does not work on NFS/VAST. At 64 nodes the lock
contention is manageable; at 128+ nodes the filesystem can't handle the
concurrent `flock()` calls.

**Fix:** Add to launch script before `srun`:

```bash
export MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron_locks
export FILELOCK_LOCK_CLASS=soft
```

`MEGATRON_CONFIG_LOCK_DIR` moves lock files to local `/tmp` (per-node,
no NFS). `FILELOCK_LOCK_CLASS=soft` uses polling instead of `fcntl` —
but only if `safe_config_loader.py` has been patched to honor it (the
upstream code hardcodes `filelock.FileLock`).

The lock only needs to coordinate processes on the same node (torchrun
spawns 4 workers per node with `--ntasks-per-node=1`), so local `/tmp`
is sufficient.

---

### OOM from Recompute Overrides

**Symptom:**

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB.
```

Crash in MLP GLU activation (`mlp.py`, `glu` function).

**Cause:** Overriding the recipe's recompute settings with:

```yaml
model:
    recompute_granularity: "full"
    recompute_method: "block"
    recompute_num_layers: 1     # <-- only recomputes 1 of N layers per stage
```

With PP=16 and 64 layers, there are 4 layers per pipeline stage.
`recompute_num_layers: 1` stores full activations for 3 of those 4
layers. The MLP activations at long sequence lengths blow up memory.

**Fix:** Do NOT override recompute. Comment out or remove the `model:`
section entirely. The recipe (`granite_30b.py`) automatically uses
selective recompute when CP > 1:

```python
if context_parallelism > 1:
    cfg.model.recompute_granularity = "selective"
else:
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    cfg.model.recompute_num_layers = 1
```

Selective recompute stores only cheap activations (norms, dropouts) and
recomputes expensive ones (attention, MLP). This is what the working
128k/64-node config used.

If you must use full recompute, set `recompute_num_layers` equal to
the layers per stage (e.g., 4 for PP=16, 8 for PP=8) to recompute
ALL layers — not just 1.

---

### OOM When Scaling Sequence Length (Same Tokens/GPU)

**Symptom:** Training works at 128k/CP=4 (32k tokens/GPU) but OOMs at
256k/CP=8 (also 32k tokens/GPU) with identical recompute settings.

**Cause:** Internal Megatron-LM and Transformer Engine buffers allocate
based on the full `model.seq_length` (262144), not `seq_length / CP`
(32768). Key sources:

- **RoPE LRU cache** (`rotary_pos_embedding.py`): cached with
  `@lru_cache(maxsize=32)`, each entry is `[max_seqlen, 1, 1, 2*dim]`
  in float32. For packed sequences, CP slicing is skipped — the full
  RoPE tensor stays on every GPU. 256k entries are 2x larger than 128k.

- **TE attention workspace**: TEDotProductAttention handles CP ring
  attention internally. Its workspace allocations depend on
  `max_seqlen_q` / `max_seqlen_kv` from `PackedSeqParams`, which are
  NOT adjusted after CP partitioning.

- **Cumulative small buffers**: position IDs, cu_seqlens, loss masks
  sized for the full pack across in-flight pipeline microbatches.

**Fix:** Increase CP to reduce tokens per GPU below 32k. For 256k,
use CP=16 (16k tokens/GPU). For 512k, use CP=32 (16k tokens/GPU).
16k tokens/GPU is the safe target for Granite 30B on GB200 (184 GB).

---

### pad_seq_to_mult Assertion Error

**Symptom:**

```
AssertionError: First dimension of the tensor should be divisible
by tensor parallel size
```

Crash in the embedding layer's `reduce_scatter_to_sequence_parallel_region`.

**Cause:** Data was packed with `pad_seq_to_mult` that is not divisible
by `TP x CP`. For example, packing with `pad_seq_to_mult=16` (using the
wrong `CP x 2` formula) and training with TP=4, CP=16 (needs 64).

After CP partitioning, each rank gets a sum of sub-sequence lengths.
Each sub-sequence is a multiple of `pad_seq_to_mult`. For the SP
reduce-scatter to work, this sum must be divisible by TP. This is only
guaranteed when `pad_seq_to_mult` is divisible by `TP x CP`.

**Fix:** Repack data with correct `pad_seq_to_mult = TP x CP`. Or pack
with `pad_seq_to_mult=256` to cover all CP values up to 64.

**Compatibility check:** Data packed with `pad_seq_to_mult=N` works for
any training config where `N % (TP x CP) == 0`:

```
pad_seq_to_mult=256 → works for TP=4 with CP=1,2,4,8,16,32,64
pad_seq_to_mult=64  → works for TP=4 with CP=1,2,4,8,16
pad_seq_to_mult=32  → works for TP=4 with CP=1,2,4,8
pad_seq_to_mult=16  → works for TP=4 with CP=1,2,4
```

---

### NCCL Hang (Training Stalls, 0% GPU Utilization)

**Symptom:** Training stops making progress. `nvitop` shows memory
allocated (e.g., 61%) but GPU utilization at 0%. No error messages in
logs. Job runs indefinitely.

**Cause:** A rank is stuck in an NCCL collective (allreduce, allgather,
etc.) due to a straggler node, network hiccup, or silent GPU error.
The default NCCL timeouts (600s) may not fire if the hang is in
PyTorch's process group layer above NCCL, or if the heartbeat thread
itself is blocked by the GIL (e.g., a VAST I/O stall).

**Distinguish from checkpoint hang:** If it happens at iteration 250,
500, etc. (multiples of `save_interval`), it's a checkpoint write stall
on VAST, not NCCL. If it happens mid-iteration (e.g., step 300), it's
NCCL.

**Fix:** Tighten timeouts and enable desync debugging:

```bash
export NCCL_TIMEOUT=300000                    # 5 min (ms), NCCL-level
export NCCL_TIMEOUT_WAIT_SEC=300              # 5 min (was 600)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=300   # 5 min (was 600)
export TORCH_NCCL_ENABLE_TIMING=1             # track collective durations
export TORCH_NCCL_DESYNC_DEBUG=1              # logs which rank is desynced
```

`TORCH_NCCL_DESYNC_DEBUG=1` is the most useful — when a timeout fires,
it tells you exactly which rank(s) are out of sync, making it easy to
identify the bad node.

The job's `cleanup` trap with `scontrol requeue` handles automatic
restart. Training resumes from the last checkpoint. One-off stragglers
typically don't recur after requeue.

---

### Checkpoint Export: lm_head.weight KeyError

**Symptom:**

```
KeyError: "Tensor 'lm_head.weight' from generator not found in the
original model structure. To ignore, set strict=False."
```

**Cause:** Granite has `tie_word_embeddings=True` — the original HF
model shares `model.embed_tokens.weight` and `lm_head.weight`. AutoBridge
unties them during training (different Granite scaling multipliers for
embedding vs logits). During export, the converter generates a separate
`lm_head.weight` that doesn't exist in the original HF model structure.

**Fix:** Add `--not-strict` to the export command. This is always
required for Granite models. The bridge correctly handles the weight
conversion and un-bakes the Granite multipliers.

---

### Triton Symlink Error on Requeued Jobs

**Symptom:**

```
ln: failed to create symbolic link '/root/.triton': File exists
```

**Cause:** `ln -s /tmp/$USER/triton ~/.triton` fails when the symlink
already exists from a previous run or requeue.

**Fix:**

```bash
ln -sfn /tmp/$USER/triton ~/.triton
```

The `-f` flag forces overwrite, `-n` treats the target as a normal file
(prevents creating a symlink inside the existing symlink target).
