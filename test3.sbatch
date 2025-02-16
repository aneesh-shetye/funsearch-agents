#!/bin/bash

#SBATCH --account pr_100_tandon_priority
#SBATCH --cpus-per-task=2
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=funsearch-imitation
#SBATCH --mail-user=axs10302@nyu.edu
#SBATCH --time=12:15:00
#SBATCH --output=stdout3.out

RUNDIR=$SCRATCH/llama3/

module purge 


singularity exec --nv --overlay /scratch/axs10302/pytorch-example3/my_pytorch.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c " source /ext3/env.sh;
torchrun funsearch run examples/test_cartpole.py 9,7 --sandbox_type ExternalProcessSandbox"


echo "Process $SLURM_PROCUD of Job $SLURM_JOBID with the local id
 $SLURM_LOCALID using gpu id $CUDA_DEVICE (we may use gpu: $CUDA_VISIBLE_DEVICES on 
$(hostname))" 
echo "computing on $(nvidia-smi --query-gpu=gpu_name --format=csv -i $CUDA_dEVICE | tail -n 1)" 
sleep 15 




