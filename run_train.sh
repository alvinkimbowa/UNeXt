#!/bin/bash

nnUNet_raw="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_raw"
nnUNet_preprocessed="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_preprocessed"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed
export NO_ALBUMENTATIONS_UPDATE=1

train=1
eval=0
analyze=1
dataset_name="Dataset073_GE_LE"
# dataset_name="Dataset072_GE_LQP9"
# dataset_name="Dataset070_Clarius_L15"
lr=0.0001
epochs=400
b=8
fold=0

gpu=1
export CUDA_VISIBLE_DEVICES=$gpu

# Evaluation settings
save_preds=false
overlay=true
largest_component=true
ckpt="model_best.pth"

# Architecture list - comment/uncomment to select which models to use
# For train/eval: uncomment only ONE architecture
# For analyze: uncomment ALL architectures you want to analyze
all_archs=(
    # "UNext"
    # "TinyUNet"
    # Exps 0: Our configuration of reduced UNet
    # "XTinyUNet"
    # "XTinyUNetB"
    # Exps 1: Semi-baseline: similar to Ulises (visual frontend)
    # "XTinyMonoUNetScale1"
    # "XTinyMonoUNetScale6"
    # "XTinyMonoV2UNetScale1"
    # "XTinyMonoV2UNetScale6"
    # Exps 2.1: Proposed: Rather than totally filtering the input, dynamically regulate the input signal using local phase info
    # "XTinyMonoV2GatedUNet"    # at visual frontend only
    # Exps 2.2: Use the same gating signal at other parts of the network. Downsample the signal to fit different resolutions.
    # "XTinyMonoV2GatedEncUNetV0"
    # "XTinyMonoV2GatedEncUNet"    # within the encoder
    # "XTinyMonoV2GatedEncUNetV1"    # within the encoder
    # "XTinyMonoV2GatedEncUNetV1B"    # within the encoder
    "XTinyMonoV2GatedEncUNetV1L"    # within the encoder
    # "XTinyMonoV2GatedEncUNetV1H"    # within the encoder
    # "XTinyMonoV2GatedEncUNetV1XL"    # within the encoder
    # "XTinyMonoV2GatedEncDecUNet"    # within the encoder and decoder
    # "XTinyMonoV2GatedEncDecUNetV1"    # within the encoder and decoder
    # "XTinyMonoV2GatedDecUNet"    # within the decoder only
    # "XTinyMonoV2GatedDecUNetV1"    # within the decoder only
    # "XTinyMonoUNetgateddec"    # within the decoder
    # Exps 2.3: Learn multi-scale gating signals - a separate layer for each stage that takes as input a downsampled version of the input image
    # "XTinyMonoUNetgatedencv1"    # within the encoder, using v1 mono layer
    # "XTinyMonoUNetgatedencdecv1"    # within the encoder and decoder, using v1 mono layer
    # "XTinyMonoUNetgateddecv1"    # within the decoder, using v1 mono layer
    # Exps 2.4: Learn multi-scale gating signals - with a strided convolution to downsample the input image
    # "XTinyMonoUNetgatedencv2"    # within the encoder, using v2 mono layer
    # "XTinyMonoUNetgatedencdecv2"    # within the encoder and decoder, using v2 mono layer
    # "XTinyMonoUNetgateddecv2"    # within the decoder, using v2 mono layer
)

# Get the first uncommented architecture for train/eval
arch="${all_archs[0]}"
if [[ $arch =~ ^[[:space:]]*# ]]; then
    echo "Error: First architecture in all_archs is commented. Please uncomment at least one architecture."
    exit 1
fi

if [[ $arch == "TinyUNet" ]]; then
    min_lr=1e-6
    loss="TinyUNetLoss"
    input_h=256
    input_w=256
    optimizer="Adam"
    scheduler="CosineAnnealingLR"
    weight_decay=1e-4
    deep_supervision=False
    data_augmentation=False
    input_channels=3
    largest_component=false
elif [[ $arch == XTiny* ]]; then
    lr=0.01
    weight_decay=0.01
    min_lr=1e-5
    loss="TinyUNetLoss"
    input_h=256
    input_w=256
    deep_supervision=False
    data_augmentation=True
    optimizer="AdamW"
    scheduler="PolyLR"
    ckpt="model_best.pth"
    input_channels=1
else
    min_lr=1e-5
    loss="BCEDiceLoss"
    input_w=512
    input_h=512
    optimizer="Adam"
    scheduler="CosineAnnealingLR"
    weight_decay=1e-4
    deep_supervision=False
    data_augmentation=False
    input_channels=3
    largest_component=false
fi

if [[ $fold -eq 5 ]]; then
    fold="all"
fi

echo "fold: $fold"
echo "train: $train"
echo "eval: $eval"
echo "analyze: $analyze"
echo "dataset_name: $dataset_name"
echo "arch: $arch"
echo "exp_name: $exp_name"
echo "lr: $lr"
echo "epochs: $epochs"
echo "input_w: $input_w"
echo "input_h: $input_h"
echo "b: $b"
echo "input_channels: $input_channels"
echo "deep_supervision: $deep_supervision"
echo "data_augmentation: $data_augmentation"

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
        --loss $loss \
        --optimizer $optimizer \
        --scheduler $scheduler \
        --weight_decay $weight_decay \
        --input_channels $input_channels \
        --deep_supervision $deep_supervision \
        --data_augmentation $data_augmentation
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
        --save_preds $save_preds \
        --ckpt $ckpt \
        --deep_supervision $deep_supervision \
        --data_augmentation $data_augmentation \
        --overlay $overlay \
        --largest_component $largest_component
    done
fi

if [[ $analyze -eq 1 ]]; then
    # Collect all uncommented architectures from all_archs array
    archs_to_analyze=()
    for a in "${all_archs[@]}"; do
        if [[ ! $a =~ ^[[:space:]]*# ]]; then
            archs_to_analyze+=("$a")
        fi
    done
    
    if [[ ${#archs_to_analyze[@]} -eq 0 ]]; then
        echo "Error: No architectures selected for analysis. Please uncomment at least one architecture in all_archs array."
        exit 1
    fi
    
    # Analyze all uncommented architectures
    for current_arch in "${archs_to_analyze[@]}"; do
        echo ""
        echo "============================================================"
        echo "Analyzing: $current_arch"
        echo "============================================================"
        
        # Determine settings based on architecture (same logic as main script)
        if [[ $current_arch == "TinyUNet" ]]; then
            analyze_input_h=256
            analyze_input_w=256
            analyze_deep_supervision=False
        elif [[ $current_arch == XTiny* ]]; then
            analyze_input_h=256
            analyze_input_w=256
            analyze_deep_supervision=False
        else
            analyze_input_w=512
            analyze_input_h=512
            analyze_deep_supervision=False
        fi
        
        analyze_args="--arch $current_arch --input_channels $input_channels --input_h $analyze_input_h --input_w $analyze_input_w --gpu $gpu --deep_supervision $analyze_deep_supervision"
        
        # Save analysis to model directory if it exists
        model_dir="models/$current_arch"
        if [[ -d "$model_dir" ]]; then
            analyze_args="$analyze_args --save $model_dir/model_analysis.json"
        fi
        
        python analyze_model.py $analyze_args
        
        echo "✓ Completed analysis for $current_arch"
    done
    
    echo ""
    echo "============================================================"
    echo "All models analyzed!"
    echo "============================================================"
fi