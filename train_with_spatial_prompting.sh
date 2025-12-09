#!/bin/bash
#
# Training script for TextBraTS with Spatial Prompting
# Uses AAL v3 expanded atlas masks (interpolated version - best performance)
#
# Expected improvements:
# - Learned alpha: 0.4-0.6 (moderate trust in masks)
# - Dice improvement: 1-3%
# - Reduced false positives in unreported regions
#

set -e  # Exit on error

# Configuration
PYTHON=/Disk1/afrouz/anaconda3/bin/python3.13
PROJECT_DIR=/Disk1/afrouz/Projects/TextBraTS
DATA_DIR=/Disk1/afrouz/Data/Merged
ATLAS_MASKS_DIR=/Disk1/afrouz/Data/TextBraTS_atlas_masks_aal_v3
TRAIN_JSON=./Train.json
VAL_JSON=./Val.json

# Training parameters
BATCH_SIZE=2
MAX_EPOCHS=200
LEARNING_RATE=1e-4
FEATURE_SIZE=48
INFER_OVERLAP=0.5

# Output directory
LOGDIR=./runs/spatial_prompting_aal_v3_$(date +%Y%m%d_%H%M%S)

echo "=========================================================================="
echo "TextBraTS Training with Spatial Prompting (AAL v3)"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  Data directory:        $DATA_DIR"
echo "  Atlas masks directory: $ATLAS_MASKS_DIR"
echo "  Training JSON:         $TRAIN_JSON"
echo "  Validation JSON:       $VAL_JSON"
echo "  Batch size:            $BATCH_SIZE"
echo "  Max epochs:            $MAX_EPOCHS"
echo "  Learning rate:         $LEARNING_RATE"
echo "  Log directory:         $LOGDIR"
echo ""
echo "Atlas Information:"
echo "  Type:                  AAL (Automated Anatomical Labeling)"
echo "  Version:               v3 expanded (with deep structures)"
echo "  GT Overlap:            43.6% (moderate, usable)"
echo "  Expected alpha:        0.4-0.6"
echo ""
echo "=========================================================================="
echo ""

# Change to project directory
cd $PROJECT_DIR

# Verify atlas masks exist
if [ ! -d "$ATLAS_MASKS_DIR" ]; then
    echo "ERROR: Atlas masks directory not found: $ATLAS_MASKS_DIR"
    exit 1
fi

# Count atlas masks
MASK_COUNT=$(ls -1 $ATLAS_MASKS_DIR/*_atlas_mask.nii.gz 2>/dev/null | wc -l)
echo "Found $MASK_COUNT atlas masks in $ATLAS_MASKS_DIR"
echo ""

# Verify training and validation JSONs exist
if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: Training JSON not found: $TRAIN_JSON"
    exit 1
fi

if [ ! -f "$VAL_JSON" ]; then
    echo "ERROR: Validation JSON not found: $VAL_JSON"
    exit 1
fi

echo "Starting training..."
echo "=========================================================================="
echo ""

# Run training with spatial prompting
$PYTHON main.py \
    --data_dir "$DATA_DIR" \
    --json_list "$TRAIN_JSON" \
    --val_json "$VAL_JSON" \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --optim_lr $LEARNING_RATE \
    --feature_size $FEATURE_SIZE \
    --roi_x 128 \
    --roi_y 128 \
    --roi_z 128 \
    --spatial_prompting \
    --atlas_masks_dir "$ATLAS_MASKS_DIR" \
    --infer_overlap $INFER_OVERLAP \
    --save_checkpoint \
    --val_every 1 \
    --logdir "$LOGDIR" \
    2>&1 | tee "${LOGDIR}_training.log"

echo ""
echo "=========================================================================="
echo "Training complete!"
echo "=========================================================================="
echo ""
echo "Results saved to: $LOGDIR"
echo "Training log:     ${LOGDIR}_training.log"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir=$LOGDIR"
echo ""
echo "To check learned alpha parameter:"
echo "  Look for 'spatial_prompt_alpha' in TensorBoard scalars"
echo "  Expected range: 0.4-0.6 (moderate trust in masks)"
echo ""
echo "=========================================================================="
