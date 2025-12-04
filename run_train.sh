#!/bin/bash

nnUNet_raw="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_raw"
nnUNet_preprocessed="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_preprocessed"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed

train=1
eval=0
dataset_name="Dataset073_GE_LE"
arch="UNext"
exp_name="Dataset073_GE_LE"
lr=0.0001
epochs=500
input_w=512
input_h=512
b=8

if [[ $train -eq 1 ]]; then
    python train.py \
        --dataset $dataset_name \
        --arch $arch \
        --name $exp_name \
        --lr $lr \
        --epochs $epochs \
        --input_w $input_w \
        --input_h $input_h \
        --b $b
fi

if [[ $eval -eq 1 ]]; then
    python val.py \
        --name $exp_name
fi