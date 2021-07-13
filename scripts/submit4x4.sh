#!/bin/bash
#SBATCH --nodes=4                           # Number of nodes
#SBATCH --gres=gpu:4                        # Number of gpus
#SBATCH --ntasks-per-node=4                 # Same as above
#SBATCH --cpus-per-task=14                  # We have 56 cpus per node so 14.
#SBATCH -o /home/fohratte/kidney_histo/logs/slurm_%j.txt      # Path to save slurm_logs (‰j is the job number).

export NCCL_DEBUG=INFO                     # Uncomment both to see actual errors that come up.
export PYTHONFAULTHANDLER=1

# $@ collects everything you pass onto the sbatch. See the example below
srun python3 -u /home/fohratte/kidneyhisto/codes/$@ --num_gpus 4 --num_nodes 4
