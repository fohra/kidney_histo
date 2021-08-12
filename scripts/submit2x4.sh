#!/bin/bash
#SBATCH --nodes=2                           # Number of nodes
#SBATCH --gres=gpu:4                        # Number of gpus
#SBATCH --ntasks-per-node=4                 # Same as above
#SBATCH --cpus-per-task=14                  # We have 56 cpus per node so 14.
#SBATCH -o /data/atte/kidney_histo/logs/slurm_%j.txt      # Path to save slurm_logs (‰j is the job number).

#export NCCL_DEBUG=INFO                     # Uncomment both to see actual errors that come up.
#export PYTHONFAULTHANDLER=1

echo Conda environent = $CONDA_DEFAULT_ENV

# $@ collects everything you pass onto the sbatch. See the example below
srun python3 -u /data/atte/kidney_histo/Classifier/$@ --num_gpus 4 --num_nodes 2
