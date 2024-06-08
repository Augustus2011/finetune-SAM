#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Define variables
arch="vit_h"  # Change this value as needed
finetune_type= "vanilla"
dataset_name="MRI-Prostate"  # Assuming you set this if it's dynamic

# Construct the checkpoint directory argument
dir_checkpoint="2D-SAM_${arch}_encoderdecoder_${finetune_type}_${dataset_name}_noprompt"
targets='combine_all' 

img_folder="/finetune-SAM/datasets/${dataset_name}/images/"  # Assuming this is the folder where images are stored
mask_folder="/finetune-SAM/datasets/${dataset_name}/masks/"

# Run the Python script
python DDP_splitgpu_train_finetune_noprompt.py \
    -if_warmup True \
    -if_split_encoder_gpus True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -sam_ckpt "sam_vit_h_4b8939.pth" \
    -img_folder "$img_folder" \
    -mask_folder "$mask_folder" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint"
