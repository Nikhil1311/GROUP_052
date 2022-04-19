#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

source ~/rl_mujoco/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py
module load python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
wandb login
cd ~/GROUP_052

python train_agent.py --group_name reset_every_100000_inter9 --step_per_inter 9 --reset_every 100000 --use_wandb --wandb_dir $SLURM_TMPDIR