# This is a PyTorch implementation of our paper.

## Train

### Pretrained ImageNet 22k weights used to initialize the method with CSWin-L 
https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_large_22k_224.pth

### Pretrained ImageNet 22k weights used to initialize the method with Volo-D5
https://github.com/sail-sg/volo/releases/download/volo_1/d5_224_86.10.pth.tar

Save it into ./checkpoints/


### CSWin-L

run: sh main.sh

OR

```bash
python main.py --model CSWin_144_24322_large_224_hydra_iccv \
--data <data path> --batch-size 16 --use-chk --img-size 224 \
--initial-checkpoint <initial checkpoint path> \
--no-resume-opt --lr-cycle-mul 1.2 --epochs 10 --warmup-epochs 0 --decay-epochs 5 --decay-rate 0.75 \
--lr-cycle-limit 4 --workers 8 --num-gpu 1 --output <output path> \
--num-classes <num classes> --cluster_file <(Food1k_clusters.csv, Food101_clusters.csv or Food251_clusters.csv)> --step <(0, 1 or 2)> \
--experiment_name <connection with Wandb, eg: Food101_Baseline> --dataset_type <(food1k, foodx251 or food101)>
```
An example can be seen in main_cswin.sh file.



### Volo-D5

run: sh main_volod5.sh

OR
```bash
python3 main.py <data path> --model volo_d5 --finetune <initial checkpoint path> --img-size 224  -b 128 \
--lr 8.0e-6 --min-lr 1.0e-6 --drop-path 0.5 --epochs 50 --apex-amp --aa augmix-m5-w4-d2 \
--experiment_name <connection with Wandb, eg: Food101_Baseline> \
--dataset_type <(food1k, foodx251 or food101)> --warmup-epochs 0 --apex-amp \
--num-classes <num classes> --cluster_file <(Food1k_clusters.csv, Food101_clusters.csv or Food251_clusters.csv)> --step <(0, 1 or 2)> 
```
An example can be seen in main.sh file.
