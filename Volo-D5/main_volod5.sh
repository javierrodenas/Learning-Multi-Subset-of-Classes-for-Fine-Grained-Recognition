#!/bin/bash
pip3 install tlt
pip3 install torchsummary
python3 main.py ./datasets/FoodX251/ --model volo_d5 --finetune ./checkpoints/d5_224_86.10.pth --img-size 224  -b 128 --lr 8.0e-6 --min-lr 1.0e-6 --drop-path 0.5 --epochs 50 --apex-amp --aa augmix-m5-w4-d2 --experiment_name "Food1k_volod5" --dataset_type "food1k" --warmup-epochs 0 --apex-amp --num-classes 1000 --cluster_file Food1k_clusters.csv --step 0

