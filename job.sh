#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

source ~/rl_mujoco/bin/activate
module load python/3.7
module load mujoco/2.0
module load mujoco-py
module load python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
wandb login
cd ~/GROUP_052

python train_agent.py --group_name hidden512_reset_every_500000_bs256_seedstep10000_inter5_lr1e-3_seed0 --step_per_inter 5 --batch_size 256 --reset_every 500000 --num_seed_steps 10000 --use_wandb --hidden_dim 512 --seed 0 --wandb_dir $SLURM_TMPDIR
