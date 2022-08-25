#! /bin/bash

# waterbirds erm
# python src/run_expt.py \
#    --dataset Waterbirds \
#    --model resnet50 \
#    --n_epochs 20 \
#    --loss erm \
#    --shift_type confounder \
#    --target_name waterbird_complete95 \
#    --confounder_names forest2water2 \
#    --show_progress


# waterbirds erm linear head only
# python src/run_expt.py \
#     --dataset Waterbirds \
#     --model linear \
#     --n_epochs 20 \
#     --loss erm \
#     --shift_type confounder \
#     --target_name waterbird_complete95 \
#     --confounder_names forest2water2 \
#     --save_best \
#     --show_progress \
#     --log_dir wb_erm_linear
    
    
# waterbirds gDRO resnet50
# CUDA_VISIBLE_DEVICES=1 python src/run_expt.py \
#     --dataset Waterbirds \
#     --model resnet50 \
#     --n_epochs 300 \
#     --weight_decay 1. \
#     --batch_size 128 \
#     --lr 0.00001 \
#     --loss group_dro \
#     --shift_type confounder \
#     --target_name waterbird_complete95 \
#     --confounder_names forest2water2 \
#     --save_best \
#     --show_progress \
#     --log_dir wb_gDRO_r50
    
    
   
# waterbirds bDRO resnet50
# CUDA_VISIBLE_DEVICES=1 python src/run_expt.py \
#     --dataset Waterbirds \
#     --model resnet50 \
#     --n_epochs 300 \
#     --weight_decay 1. \
#     --batch_size 128 \
#     --lr 0.00001 \
#     --loss bitrate_dro \
#     --shift_type confounder \
#     --target_name waterbird_complete95 \
#     --confounder_names forest2water2 \
#     --save_best \
#     --show_progress \
#     --log_dir wb_bDRO_r50


# celeba gDRO resnet50
# CUDA_VISIBLE_DEVICES=1 python src/run_expt.py \
#     --dataset CelebA \
#     --fraction 0.1 \
#     --metadata_csv_name metadata.csv \
#     --model resnet50 \
#     --n_epochs 50 \
#     --weight_decay 0.1 \
#     --batch_size 128 \
#     --lr 0.00001 \
#     --loss group_dro \
#     --reweight_groups \
#     --shift_type confounder \
#     --target_name Blond_Hair \
#     --confounder_names Male \
#     --save_best \
#     --show_progress \
#     --log_dir celeba0.1_gDRO_r50

# celeba bDRO resnet50
CUDA_VISIBLE_DEVICES=0 python src/run_expt.py \
    --dataset CelebA \
    --fraction 0.1 \
    --metadata_csv_name metadata.csv \
    --model resnet50 \
    --n_epochs 50 \
    --weight_decay 0.1 \
    --batch_size 128 \
    --lr 0.00001 \
    --loss bitrate_dro \
    --reweight_samples \
    --shift_type confounder \
    --target_name Blond_Hair \
    --confounder_names Male \
    --show_progress \
    --log_dir celeba_bDRO_r50