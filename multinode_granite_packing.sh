#!/bin/bash
#SBATCH --partition=hpc-mid
#SBATCH --nodes=4
#SBATCH --job-name=granite-pack
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=144
#SBATCH --output="/mnt/ss/proj/checkpoints/bathen/logs/granite-pack-out.%j.log"
#SBATCH --error="/mnt/ss/proj/checkpoints/bathen/logs/granite-pack-err.%j.log"
#SBATCH --wait-all-nodes=1
#SBATCH --mem=0

# =============================================================================
# Multi-node Ray data packing — Granite 4.2 SFT (7M balanced blend)
#
# Packs ~28M available samples into 128K-token bins with pad_seq_to_mult=256
# for CP flexibility. Blend weights control training-time sampling to ~7M.
#
# Container constraint: Ray head + data prep must run in ONE srun so the
# container (and Ray daemon) stays alive for the duration of the job.
#
# Usage:
#   sbatch multi_node_granite_packing.sh
#   sbatch --nodes=8 multi_node_granite_packing.sh
#   CFG=other.yaml sbatch multi_node_granite_packing.sh
# =============================================================================

. ~/.bashrc
source ~/run.env

# ---------- Config -----------------------------------------------------------
: "${CFG:=tokenization_128k_granite_8b_pad256.yaml}"
BASE_PATH=src/nemotron/recipes/granite30/stage1_sft/config/data_prep

echo "$(date) Config: ${CFG}"
echo "$(date) Nodes: ${SLURM_NNODES} x ${SLURM_CPUS_PER_TASK} CPUs = $((SLURM_NNODES * SLURM_CPUS_PER_TASK)) total CPUs"

# ---------- Container --------------------------------------------------------
container_image="/mnt/ss/squash/nemo_sft_0430.sqsh"
container_mounts="/mnt/ss:/mnt/vast,/tmp:/tmp"

SRUN_ARGS="--kill-on-bad-exit=1 \
            --container-image=${container_image} \
            --container-mounts=${container_mounts} \
            --no-container-remap-root \
            --container-workdir=/mnt/home/bathen/src/github.com/Nemotron"

# ---------- Ray env ----------------------------------------------------------
export RAY_MAX_LIMIT_FROM_API_SERVER=40000
export RAY_MAX_LIMIT_FROM_DATA_SOURCE=40000

RAY_TMPDIR="/tmp/ray_granite_pack_${SLURM_JOBID}"
echo "$(date) Ray temp dir: $RAY_TMPDIR"

# ---------- Discover nodes ---------------------------------------------------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
MASTER_ADDR="${head_node}"
MASTER_PORT=6379
ip_head="${MASTER_ADDR}:${MASTER_PORT}"
worker_num=$((SLURM_JOB_NUM_NODES - 1))

echo "$(date) Head: $head_node | Workers: $worker_num | ip_head: $ip_head"

# ---------- Start Ray workers (background, all in parallel) -----------------
# Workers use --block to keep their containers alive. They retry connecting
# to the head until it appears (Ray handles this internally).
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "$(date) Starting Ray WORKER $i on $node_i"
    srun ${SRUN_ARGS} --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "mkdir -p ${RAY_TMPDIR} && \
                 ray stop --force 2>/dev/null; \
                 ray start \
                     --address '$ip_head' \
                     --num-cpus=${SLURM_CPUS_PER_TASK} \
                     --num-gpus=0 \
                     --temp-dir '${RAY_TMPDIR}' \
                     --block" &
done

# ---------- Start Ray head + run data prep (single srun) --------------------
# Head and data prep MUST share one srun so the container stays alive.
# Workers connect once the head binds the port.
echo "$(date) Starting Ray HEAD + data prep on $head_node"
srun ${SRUN_ARGS} --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "mkdir -p ${RAY_TMPDIR} && \
             ray stop --force 2>/dev/null; \
             ray start --head \
                 --node-ip-address='$MASTER_ADDR' \
                 --port=$MASTER_PORT \
                 --num-cpus=${SLURM_CPUS_PER_TASK} \
                 --num-gpus=0 \
                 --temp-dir '${RAY_TMPDIR}' \
                 --include-dashboard=True \
                 --dashboard-host=0.0.0.0; \
             export RAY_ADDRESS='${ip_head}'; \
             sleep 15; \
             echo 'Ray cluster status:'; \
             ray status; \
             echo 'Starting data prep pipeline...'; \
             python src/nemotron/recipes/super3/stage1_sft/data_prep.py \
                 --config ${BASE_PATH}/${CFG}; \
             rc=\$?; \
             echo '=== Ray dashboard log (last 50 lines) ==='; \
             tail -50 ${RAY_TMPDIR}/session_latest/logs/dashboard.log 2>/dev/null || echo 'No dashboard log found'; \
             ray stop --force; \
             exit \$rc"

rc=$?
echo "$(date) Data prep finished with rc=$rc"
exit $rc
