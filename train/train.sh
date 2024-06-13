#!/bin/bash
#SBATCH   --job-name=PlanText-Train
#SBATCH   --partition=gpu-a40
#SBATCH   --output=log/%j.out
#SBATCH   --error=log/%j..err
#SBATCH   -c 5
#SBATCH   --gres=gpu:1
source activate mlc
python -u main.py --model_path=output18 --batch_size 32 --pre_epoch 20 --debug