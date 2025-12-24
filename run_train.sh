#!/bin/bash

nnUNet_raw="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_raw"
nnUNet_preprocessed="/home/ultrai/UltrAi/knee_us_segmentation/data/nnUNet_preprocessed"

export nnUNet_raw=$nnUNet_raw
export nnUNet_preprocessed=$nnUNet_preprocessed
export NO_ALBUMENTATIONS_UPDATE=1

train=1
eval=1
analyze=0
cascade_refiner=true
refiner_suffix="refiner"
mask_dropout=0.0
mask_dropout_foreground_only=true
save_mask_debug=true
mask_debug_samples=1
mask_debug_every=1
mask_patch_prob=0.7
mask_patch_empty_prob=0.1
mask_patch_bands=4
mask_patch_min_bands=1
mask_patch_max_bands=3
mask_foreground_prob=0.1
mask_foreground_blobs_min=1
mask_foreground_blobs_max=1
mask_foreground_radius_min=6
mask_foreground_radius_max=24
mask_shift_prob=0.5
mask_shift_max=16
mask_rotate_prob=0.5
mask_rotate_max_deg=10
model_dir_suffix=""
if [[ $cascade_refiner == True || $cascade_refiner == true ]]; then
    model_dir_suffix="$refiner_suffix"
fi
# dataset_name="Dataset072_GE_LQP9"
dataset_name="Dataset073_GE_LE"
# dataset_name="Dataset070_Clarius_L15"
# dataset_name="Dataset050_Tufts_preop_Harkey"
lr=0.0001
epochs=500
b=8
fold=0
resume_ckpt="auto"   # "auto" will use models/<arch_name>/<dataset>/fold_<fold>/checkpoint_latest.pth

gpu=1
export CUDA_VISIBLE_DEVICES=$gpu

# Evaluation settings
save_preds=false
overlay=true
data_augmentation=false
largest_component=true
test_datasets=("Dataset072_GE_LQP9" "Dataset073_GE_LE" "Dataset070_Clarius_L15" "Dataset078_KneeUS_OtherDevices")
# test_datasets=("Dataset073_GE_LE" "Dataset070_Clarius_L15" "Dataset078_KneeUS_OtherDevices")
# test_datasets=("Dataset072_GE_LQP9" "Dataset070_Clarius_L15" "Dataset078_KneeUS_OtherDevices")
# test_datasets=("Dataset072_GE_LQP9" "Dataset073_GE_LE" "Dataset078_KneeUS_OtherDevices")
# test_datasets=("Dataset073_GE_LE")
# test_datasets=("Dataset078_KneeUS_OtherDevices")
# test_datasets=("Dataset079_KneeUS_Ilker")
ckpt="model_latest.pth"

# Architecture list - comment/uncomment to select which models to use
# For train/eval: uncomment only ONE architecture
# For analyze: uncomment ALL architectures you want to analyze
all_archs=(
    # "MonoUNetBase"
    # "MonoUNetBase"
    # "MonoUNetE1"
    "MonoUNetE12"
    # "MonoUNetE123"
    # "MonoUNetE1234"
    # "MonoUNetE1234D1"
    # "MonoUNetE1234D12"
    # "MonoUNetE1234D123"
    # "UNext"
    # "TinyUNet"
    # Exps 0: Our configuration of reduced UNet
    # "XTinyUNet"
    # "XTinyUNetB"
    # "XTinyUNetL"
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
    # "XTinyMonoV3GatedEncUNetV1"    # within the encoder
    # "XTinyMonoV4GatedEncUNetV1"    # within the encoder
    # "XTinyMonoV2GatedEncUNetV1B"    # within the encoder
    # "XTinyMonoV2GatedEncUNetV1L"    # within the encoder
    # "XTinyMonoV2GatedEncUNetV1LV3"    # within the encoder
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

if [[ $arch == "UNet" ]]; then
    min_lr=1e-5
    loss="UNetLoss"
    input_h=256
    input_w=256
    optimizer="SGD"
    momentum=0.99
    scheduler="ConstantLR"
    weight_decay=1e-4
    input_channels=1
    deep_supervision=False
    num_classes=2
elif [[ $arch == "TinyUNet" ]]; then
    min_lr=1e-6
    loss="TinyUNetLoss"
    input_h=256
    input_w=256
    optimizer="Adam"
    scheduler="CosineAnnealingLR"
    weight_decay=1e-4
    deep_supervision=False
    input_channels=3
    num_classes=1
elif [[ $arch == XTiny* || $arch == MonoUNet* ]]; then
    lr=0.01
    weight_decay=0.01
    min_lr=1e-5
    loss="TinyUNetLoss"
    input_h=256
    input_w=256
    deep_supervision=False
    optimizer="AdamW"
    scheduler="PolyLR"
    input_channels=1
    num_classes=1
else
    min_lr=1e-5
    loss="BCEDiceLoss"
    input_w=256
    input_h=256
    optimizer="Adam"
    scheduler="CosineAnnealingLR"
    weight_decay=1e-4
    deep_supervision=False
    input_channels=3
    num_classes=1
fi

if [[ $fold -eq 5 ]]; then
    fold="all"
fi


echo "gpu: $gpu"
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
echo "largest_component: $largest_component"
echo "num_classes: $num_classes"

if [[ $train -eq 1 ]]; then
    arch_name="$arch"
    if [[ $deep_supervision == True || $deep_supervision == true ]]; then
        arch_name="${arch_name}DS"
    fi
    if [[ $data_augmentation == True || $data_augmentation == true ]]; then
        arch_name="${arch_name}DA"
    fi
    refiner_arch_name="$arch_name"
    if [[ $cascade_refiner == True || $cascade_refiner == true ]]; then
        refiner_arch_name="${arch_name}_${refiner_suffix}"
    fi

    if [[ "$resume_ckpt" == "auto" || -z "$resume_ckpt" ]]; then
        # Search for either model_latest.pth (for resuming) or model_final.pth (if training completed)
        ckpt_dir="models/${refiner_arch_name}/${dataset_name}/fold_${fold}"
        if [[ -f "${ckpt_dir}/model_latest.pth" ]]; then
            resume_ckpt="${ckpt_dir}/model_latest.pth"
        elif [[ -f "${ckpt_dir}/model_final.pth" ]]; then
            resume_ckpt="${ckpt_dir}/model_final.pth"
        else
            resume_ckpt="${ckpt_dir}/model_latest.pth"  # Default path, will be created if doesn't exist
        fi
    fi

    python train.py \
        --dataset $dataset_name \
        --arch $arch \
        --cascade_refiner $cascade_refiner \
        --base_arch $arch \
        --base_ckpt "models/${arch_name}/${dataset_name}/fold_${fold}/${ckpt}" \
        --model_dir_suffix $refiner_suffix \
        --mask_dropout $mask_dropout \
        --mask_dropout_foreground_only $mask_dropout_foreground_only \
        --mask_patch_prob $mask_patch_prob \
        --mask_patch_empty_prob $mask_patch_empty_prob \
        --mask_patch_bands $mask_patch_bands \
        --mask_patch_min_bands $mask_patch_min_bands \
        --mask_patch_max_bands $mask_patch_max_bands \
        --mask_foreground_prob $mask_foreground_prob \
        --mask_foreground_blobs_min $mask_foreground_blobs_min \
        --mask_foreground_blobs_max $mask_foreground_blobs_max \
        --mask_foreground_radius_min $mask_foreground_radius_min \
        --mask_foreground_radius_max $mask_foreground_radius_max \
        --mask_shift_prob $mask_shift_prob \
        --mask_shift_max $mask_shift_max \
        --mask_rotate_prob $mask_rotate_prob \
        --mask_rotate_max_deg $mask_rotate_max_deg \
        --save_mask_debug $save_mask_debug \
        --mask_debug_samples $mask_debug_samples \
        --mask_debug_every $mask_debug_every \
        --lr $lr \
        --epochs $epochs \
        --input_w $input_w \
        --input_h $input_h \
        -b $b \
        --fold $fold \
        --min_lr $min_lr \
        --loss $loss \
        --optimizer $optimizer \
        --scheduler $scheduler \
        --weight_decay $weight_decay \
        --input_channels $input_channels \
        --deep_supervision $deep_supervision \
        --data_augmentation $data_augmentation \
        --num_classes $num_classes \
        --resume "$resume_ckpt"
fi

if [[ $eval -eq 1 ]]; then
    for test_dataset in ${test_datasets[@]}; do
        echo "Evaluating $test_dataset"
        if [[ $test_dataset == "Dataset078_KneeUS_OtherDevices" || $test_dataset == "Dataset079_KneeUS_Ilker" ]]; then
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
        --largest_component $largest_component \
        --model_dir_suffix $model_dir_suffix
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
        elif [[ $current_arch == XTiny* || $current_arch == MonoUNet* ]]; then
            analyze_input_h=256
            analyze_input_w=256
            analyze_deep_supervision=False
        else
            analyze_input_w=512
            analyze_input_h=512
            analyze_deep_supervision=False
        fi
        
        analyze_args="--arch $current_arch --input_channels $input_channels --input_h $analyze_input_h --input_w $analyze_input_w --gpu $gpu --deep_supervision $analyze_deep_supervision"
        if [[ $cascade_refiner == True || $cascade_refiner == true ]]; then
            analyze_args="$analyze_args --cascade_refiner $cascade_refiner --base_arch $current_arch --base_ckpt models/${current_arch}/${dataset_name}/fold_${fold}/${ckpt} --mask_dropout $mask_dropout --mask_dropout_foreground_only $mask_dropout_foreground_only --mask_patch_prob $mask_patch_prob --mask_patch_empty_prob $mask_patch_empty_prob --mask_patch_bands $mask_patch_bands --mask_patch_min_bands $mask_patch_min_bands --mask_patch_max_bands $mask_patch_max_bands --mask_foreground_prob $mask_foreground_prob --mask_foreground_blobs_min $mask_foreground_blobs_min --mask_foreground_blobs_max $mask_foreground_blobs_max --mask_foreground_radius_min $mask_foreground_radius_min --mask_foreground_radius_max $mask_foreground_radius_max --mask_shift_prob $mask_shift_prob --mask_shift_max $mask_shift_max --mask_rotate_prob $mask_rotate_prob --mask_rotate_max_deg $mask_rotate_max_deg"
        fi
        
        # Save analysis to model directory if it exists
        if [[ $data_augmentation == true ]]; then
            current_arch="$current_arch"DA
        fi

        model_dir="models/$current_arch"
        if [[ $cascade_refiner == True || $cascade_refiner == true ]]; then
            model_dir="models/${current_arch}_${refiner_suffix}"
        fi
        echo "model_dir: $model_dir"
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
