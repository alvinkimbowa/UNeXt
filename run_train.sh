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
epochs=400
input_w=512
input_h=512
b=8
fold=0

# Evaluation settings
test_datasets=("Dataset073_GE_LE" "Dataset072_GE_LQP9" "Dataset070_Clarius_L15" "Dataset078_KneeUS_OtherDevices")
test_split="Tr"

if [[ $train -eq 1 ]]; then
    python train.py \
        --dataset $dataset_name \
        --arch $arch \
        --lr $lr \
        --epochs $epochs \
        --input_w $input_w \
        --input_h $input_h \
        --b $b \
        --fold $fold
fi

if [[ $eval -eq 1 ]]; then
    for test_dataset in ${test_datasets[@]}; do
        echo "Evaluating $test_dataset"
        if [[ $test_dataset == "Dataset078_KneeUS_OtherDevices" ]]; then
            test_split="Ts"
        else
            test_split="Tr"
        fi
        python val.py \
        --train_dataset $dataset_name \
        --train_fold $fold \
        --test_dataset $test_dataset \
        --test_split $test_split
    done
fi