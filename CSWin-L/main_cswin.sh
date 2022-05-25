#!/bin/bash

pip install torchsummary
pip install einops

# CUDA_VISIBLE_DEVICES=2
python main.py --model CSWin_144_24322_large_224_hydra_iccv \
--data ./datasets/FoodX251/ --batch-size 16 --use-chk --img-size 224 \
--dataset_type foodx251 --initial-checkpoint ./checkpoints/cswin_large_22k_224.pth \
--no-resume-opt --lr-cycle-mul 1.2 --epochs 10 --warmup-epochs 0 --decay-epochs 5 --decay-rate 0.75 \
--lr-cycle-limit 4 --workers 8 --num-gpu 1 --output /media/HDD_4TB/HDD_4TB_1/Checkpoints/FoodX251/CSWIN/ \
--experiment_name "FoodX251_Baseline_GradientAccumulation" --gradient_accumulation 10 --step 0