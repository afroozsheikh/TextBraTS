#!/bin/bash
#
# Baseline training script for TextBraTS WITHOUT Spatial Prompting
# Use this to compare against spatial prompting results
#

set -e  # Exit on error

# Configuration
PYTHON=/Disk1/afrouz/anaconda3/bin/python3.13
PROJECT_DIR=/Disk1/afrouz/Projects/TextBraTS
DATA_DIR=/Disk1/afrouz/Data/Merged
TRAIN_JSON=./Train.json
VAL_JSON=./Val.json

# Training parameters (same as spatial prompting for fair comparison)
BATCH_SIZE=2
MAX_EPOCHS=200
LEARNING_RATE=1e-4
FEATURE_SIZE=48
INFER_OVERLAP=0.5

# Output directory
LOGDIR=./runs/baseline_no_spatial_prompting_$(date +%Y%m%d_%H%M%S)

echo "=========================================================================="
echo "TextBraTS Baseline Training (NO Spatial Prompting)"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  Data directory:        $DATA_DIR"
echo "  Training JSON:         $TRAIN_JSON"
echo "  Validation JSON:       $VAL_JSON"
echo "  Batch size:            $BATCH_SIZE"
echo "  Max epochs:            $MAX_EPOCHS"
echo "  Learning rate:         $LEARNING_RATE"
echo "  Log directory:         $LOGDIR"
echo ""
echo "Note: This is the BASELINE without spatial prompting"
echo "      Compare results with spatial prompting experiment"
echo ""
echo "=========================================================================="
echo ""

# Change to project directory
cd $PROJECT_DIR

# Verify training and validation JSONs exist
if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: Training JSON not found: $TRAIN_JSON"
    exit 1
fi

if [ ! -f "$VAL_JSON" ]; then
    echo "ERROR: Validation JSON not found: $VAL_JSON"
    exit 1
fi

echo "Starting baseline training..."
echo "=========================================================================="
echo ""

# Run training WITHOUT spatial prompting
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
    --infer_overlap $INFER_OVERLAP \
    --save_checkpoint \
    --val_every 1 \
    --logdir "$LOGDIR" \
    2>&1 | tee "${LOGDIR}_training.log"

echo ""
echo "=========================================================================="
echo "Baseline training complete!"
echo "=========================================================================="
echo ""
echo "Results saved to: $LOGDIR"
echo "Training log:     ${LOGDIR}_training.log"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir=$LOGDIR"
echo ""
echo "=========================================================================="
