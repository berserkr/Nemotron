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

# ---------- Ray state API limits (match ray_cpu.sub.j2 template) -------------
export RAY_MAX_LIMIT_FROM_API_SERVER=40000
export RAY_MAX_LIMIT_FROM_DATA_SOURCE=40000

# ---------- Ray temp dir (on shared /tmp, visible across containers) ---------
RAY_TMPDIR="/tmp/ray_dataprep_${SLURM_JOBID}"
echo "$(date) Ray temp dir: $RAY_TMPDIR"

# ---------- Discover nodes ---------------------------------------------------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
MASTER_PORT=6379
ip_head="${MASTER_ADDR}:${MASTER_PORT}"

echo "$(date) Head node: $head_node"
echo "$(date) MASTER_ADDR: $MASTER_ADDR"
echo "$(date) ip_head: $ip_head"
echo "$(date) All nodes: $nodes"

# ---------- Start Ray workers (background, before head) ---------------------
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "$(date) Starting Ray WORKER $i on $node_i"
    srun ${SRUN_ARGS} --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "mkdir -p ${RAY_TMPDIR} && \
                 ray stop --force 2>/dev/null; sleep 3; \
                 ray start \
                     --address '$ip_head' \
                     --num-cpus=${SLURM_CPUS_PER_TASK} \
                     --temp-dir '${RAY_TMPDIR}' \
                     --block" &
    sleep 6
done

# ---------- Start Ray head + run data prep (same container) ------------------
# Running data prep in the same container as the Ray head so the dashboard
# is accessible on localhost:8265 (cosmos_xenna monitoring needs it).
echo "$(date) Starting Ray HEAD + data prep on $head_node"
srun ${SRUN_ARGS} --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "mkdir -p ${RAY_TMPDIR} && \
             ray stop --force 2>/dev/null; sleep 3; \
             ray start --head \
                 --node-ip-address='$MASTER_ADDR' \
                 --port=$MASTER_PORT \
                 --num-cpus=${SLURM_CPUS_PER_TASK} \
                 --temp-dir '${RAY_TMPDIR}' \
                 --include-dashboard=True \
                 --dashboard-host=0.0.0.0; \
             sleep 15; \
             echo 'Ray cluster status:'; \
             ray status; \
             echo 'Starting data prep pipeline...'; \
             python src/nemotron/recipes/super3/stage1_sft/data_prep.py \
                 --config ${BASE_PATH}/${CFG}; \
             rc=\$?; \
             echo '=== Ray dashboard log ==='; \
             cat ${RAY_TMPDIR}/session_latest/logs/dashboard.log 2>/dev/null || echo 'No dashboard log found'; \
             exit \$rc"

rc=$?
echo "$(date) Data prep finished with rc=$rc"
exit $rc
