#!/bin/bash

nnUNet_raw="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_raw"
nnUNet_preprocessed="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_preprocessed"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed
export NO_ALBUMENTATIONS_UPDATE=1

train=1
eval=0
dataset_name="Dataset073_GE_LE"
arch="UNext"
lr=0.0001
epochs=400
b=8
fold=4
save_preds=true

if [[ $arch == "UNext" || $arch == "UNext_S" ]]; then
    min_lr=1e-5
    loss="BCEDiceLoss"
    input_w=512
    input_h=512
else
    min_lr=1e-6
    loss="TinyUNetLoss"
    input_h=256
    input_w=256
fi

echo "arch: $arch"

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
        --fold $fold \
        --min_lr $min_lr \
        --loss $loss
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
        --name $arch \
        --train_dataset $dataset_name \
        --train_fold $fold \
        --test_dataset $test_dataset \
        --test_split $test_split \
        --save_preds $save_preds
    done
fi