#!/bin/bash
#SBATCH --partition=hpc-mid
#SBATCH --nodes=4
#SBATCH --job-name=data-pack-ray
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=144
#SBATCH --output="/mnt/vast/proj/checkpoints/bathen/logs/dataprep-out.%j.log"
#SBATCH --error="/mnt/vast/proj/checkpoints/bathen/logs/dataprep-err.%j.log"
#SBATCH --wait-all-nodes=1
#SBATCH --mem=0

# =============================================================================
# Multi-node Ray data packing for Nemotron SFT
#
# Usage:
#   sbatch data_packing_multinode.sh                          # uses default CFG
#   sbatch --nodes=8 data_packing_multinode.sh                # scale to 8 nodes
#   CFG=test_granite_3b_math_128k_cp2.yaml sbatch data_packing_multinode.sh
# =============================================================================

. ~/.bashrc
source ~/run.env

# ---------- Config -----------------------------------------------------------
: "${CFG:=test_granite_3b_math_128k_cp2.yaml}"
BASE_PATH=src/nemotron/recipes/super3/stage1_sft/config/data_prep

echo "$(date) Data prep config: ${CFG}"
echo "$(date) Nodes: ${SLURM_NNODES}"

# ---------- Container --------------------------------------------------------
container_image="/mnt/vast/squash/nemo_sft_python312_v4.sqsh"
container_mounts="/mnt:/mnt,/tmp:/tmp"

SRUN_ARGS="--kill-on-bad-exit=1 \
            --container-image=${container_image} \
            --container-mounts=${container_mounts} \
            --no-container-remap-root \
            --container-workdir=/mnt/home/bathen/src/github.com/Nemotron"

# ---------- Ray temp dir (shared across nodes via /tmp) ----------------------
unique_dir=$(mktemp -d /tmp/ray_dataprep_XXXX)
export RAY_TMPDIR=$unique_dir
echo "$(date) Ray temp dir: $unique_dir"

# ---------- Discover nodes ---------------------------------------------------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_PORT=6379
ip_head="${head_node}:${MASTER_PORT}"

echo "$(date) Head node: $head_node"
echo "$(date) All nodes: $nodes"

# ---------- Start Ray head ---------------------------------------------------
echo "$(date) Starting Ray HEAD on $head_node"
srun ${SRUN_ARGS} --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "ray stop --force 2>/dev/null; sleep 3; \
             ray start --head \
                 --node-ip-address='$head_node' \
                 --port=$MASTER_PORT \
                 --num-cpus=${SLURM_CPUS_PER_TASK} \
                 --temp-dir '$unique_dir' \
                 --block" &

echo "$(date) Waiting 15s for Ray head to initialize..."
sleep 15

# ---------- Start Ray workers ------------------------------------------------
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "$(date) Starting Ray WORKER $i on $node_i"
    srun ${SRUN_ARGS} --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "ray stop --force 2>/dev/null; sleep 3; \
                 ray start \
                     --address '$ip_head' \
                     --num-cpus=${SLURM_CPUS_PER_TASK} \
                     --temp-dir '$unique_dir' \
                     --block" &
    sleep 6
done

echo "$(date) Waiting 10s for all workers to join..."
sleep 10

# ---------- Run data prep on head node ---------------------------------------
echo "$(date) Starting data prep pipeline: ${CFG}"
srun --overlap -w "$head_node" --ntasks=1 --nodes=1 ${SRUN_ARGS} \
    bash -c "export RAY_ADDRESS='${ip_head}'; \
             python src/nemotron/recipes/super3/stage1_sft/data_prep.py \
                 --config ${BASE_PATH}/${CFG}"

rc=$?
echo "$(date) Data prep finished with rc=$rc"
exit $rc
