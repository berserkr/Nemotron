#!/bin/bash
#SBATCH --partition=hpc-mid
#SBATCH --nodes=16
#SBATCH --job-name=granite-nemotron-phase1
#SBATCH --ntasks-per-node=1  #<--must be 1 for torchrun / override for others like mpi
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=144 
#SBATCH --output="/mnt/vast/proj/checkpoints/bathen/logs/nemotron-out.%j.log" 
#SBATCH --error="/mnt/vast/proj/checkpoints/bathen/logs/nemotron-err.%j.log" 
#SBATCH --open-mode=append    #<--- for a requeued job so it does not wipe out logs
#SBATCH --wait-all-nodes=1
#SBATCH --mem=0
#SBATCH --segment=2 # changed form 2-->8


. ~/.bashrc
source ~/run.env

min_bw=$( echo "140" | bc -l)
sleep="300s"

#weights and biases
export WANDB_BASE_URL=https://wandb.vpc.res.ibm.com
export WANDB_ENTITY=bathen
export WANDB_PROJECT=super3-models-sft
export WANDB_DISABLE_CODE=1
export WANDB_DISABLE_GIT=1
export WANDB__SERVICE_WAIT=300

: "${PREFLIGHT_TEST:=0}"
: "${SLURM_RESTART_COUNT:=0}" 
MAX_SLURM_RESTART_COUNT=4 #change as you see fit

#trap any exit and requeue if needed 
function cleanup {
    local exit_status=$?
    if [ "$exit_status" -ne 0 ]; then
      if [ $SLURM_RESTART_COUNT -gt $MAX_SLURM_RESTART_COUNT ]; then
        echo -e "\n$(date) ${SLURM_JOB_ID} SLURM_RESTART_COUNT exceeded ${SLURM_RESTART_COUNT} .. exiting"
        exit ${SLURM_RESTART_COUNT}
      fi
      echo -e "\n$(date) ${SLURM_JOB_ID} Requeued"
      scontrol requeue ${SLURM_JOB_ID}
    else
      echo "Script exiting with status: $exit_status"
    fi
}

trap cleanup EXIT

##pre-flight NCCL Perf test on all GPUs
if [ ${PREFLIGHT_TEST} -gt 0 ]; then
    set +e
    #use the compute node's nccl test and not container 
    nccl_test="alltoall_perf"
    echo -e "$(date) ${SLURM_JOBID} Pre Filght NCCL Test  ${nccl_test}" 
    srun --ntasks-per-node=1 --mpi=pmix /opt/nccl-tests/build/${nccl_test} -T 120 -b 1G -e 8G -f 2 -g 4
    rc=$?
    if [ $rc -ne 0 ]; then
      echo -e "\n$(date) ${SLURM_JOBID} NCCL test failed .. Exiting $rc" 
      exit $rc
    fi
    set -e
    echo "$(date) Pre-flight test passed" 
fi 
#end pre-flight test 

echo "$(date) SLURM_RESTART_COUNT == ${SLURM_RESTART_COUNT}"

PYXIS_DEFAULTS=( '--no-container-mount-home' '--no-container-remap-root')

container_mounts="/mnt:/mnt"
container_image="/mnt/vast/squash/nemo_sft_python312_v4.sqsh"
LOG=/mnt/vast/proj/checkpoints/bathen/logs/${SHORT_NAME}_${SLURM_JOBID}.log

#default nccl vars handled in .nccl.conf
export TOKENIZERS_PARALLELISM=false 
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0
export NVIDIA_IMEX_CHANNELS=0
export NCCL_NVLS_ENABLE=0
export PMIX_MCA_gds='^ds12'
export NCCL_MIN_CTAS=32
export NCCL_NET_GDR_C2C=1
export NCCL_WORK_FIFO_DEPTH=1048576
export NCCL_TIMEOUT_WAIT_SEC=600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

#add some new Torch NCCL directives for debugging 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export PYTHONUNBUFFERED=TRUE 
export OMP_NUM_THREADS=64  

export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"
export MASTER_PORT=28444
export NNODES=$SLURM_NNODES
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export JOB_ID=${SLURM_JOBID}

export NCCL_DEBUG=WARN 
export NCCL_DEBUG_SUBSYS=BOOTSTRAP,INIT,NET,ENV
export NCCL_DEBUG_FILE=${NCCL_LOGS_PATH}/NCCL_DEBUG_FILE.%h.txt

export TORCH_CUDA_ARCH_LIST="Blackwell" # 12.0+PTX
export CUTE_ARCH_LDSM_SM100A_ENABLED=1 
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
export TORCHINDUCTOR_REORDER_FOR_PEAK_MEMORY=1

mkdir -p /tmp/$USER/triton
export TRITON_HOME=/tmp/$USER/triton
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
ln -s /tmp/$USER/triton ~/.triton

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

# NOTE STRUGGLER: I left your workdir intact, but if your src/nemotron/recipes... 
# path lives somewhere else, ensure this points to the correct root directory!
SRUN_ARGS="--kill-on-bad-exit=1  \
            --container-image=${container_image}  \
            --container-mounts=${container_mounts}  \
            --no-container-remap-root \
            --container-workdir=/mnt/home/bathen/src/github.com/Nemotron
            "
echo $SRUN_ARGS
echo `date` : ${SLURM_JOBID}

# --- NEW TORCHRUN MULTI-NODE DISTRIBUTED ARGS ---
export DISTRIBUTED_ARGS=" \
    --nnodes ${NNODES} \
    --nproc_per_node ${GPUS_PER_NODE} \
    --node_rank \$SLURM_NODEID \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    "
echo $DISTRIBUTED_ARGS

# --- THE FINAL STRIKE: NEMOTRON EXECUTION ---
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_HOME=/usr/local/cuda-12 torchrun ${DISTRIBUTED_ARGS} src/nemotron/recipes/super3/stage1_sft/train.py --config src/nemotron/recipes/super3/stage1_sft/config/train_128k_qwen.yaml"

echo "*********************** START ****************************"
echo $CMD

srun ${SRUN_ARGS} bash -c "${CMD}"
rc=$?
echo "rc=${rc}"
sacct -j $SLURM_JOBID -o "jobid,jobname,start,end,state"
exit $rc