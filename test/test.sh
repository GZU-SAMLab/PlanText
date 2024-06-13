#!/bin/bash
#SBATCH   --job-name=PlanText-Test
#SBATCH   --partition=gpu-a10
#SBATCH   --output=log/%j.out
#SBATCH   --error=log/%j..err
#SBATCH   -c 5
#SBATCH   --gres=gpu:1
source activate mlc
python -u main2.py --batch_size 1 --test --test_epoch 20 --model_path=output18 --batch_size 1
