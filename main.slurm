#!/bin/bash
#SBATCH -J TR-for
#SBATCH -p kshdexclu04
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=dcu:4
#SBATCH -N 2
##SBATCH -x
base_log_dir=./log/$SLURM_JOB_ID
mkdir -p ./log
mkdir -p $base_log_dir
#SBATCH -o $base_log_dir/j.o  
#SBATCH -e $base_log_dir/j.e 

mkdir -p $base_log_dir/output

dmesg_log=$base_log_dir/dmesg
debug_log=$base_log_dir/debug
output_log=$base_log_dir/output


export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
NODE_RANK=$SLURM_NODEID


# activate conda environment
source activate TRCV

module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module switch compiler/rocm/dtk-23.04


srun torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node 4 \
    --node_rank $NODE_RANK \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    main.py --output_log $output_log
