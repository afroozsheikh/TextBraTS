# Attention Heatmap Visualization

This directory contains scripts to visualize attention heatmaps from the TextSwinUNETR model, similar to Figure 5 in the paper.

## Overview

The visualization shows:
- **Before Fusion (enc3)**: Features before text-image fusion (192 channels, 16×16×16 spatial resolution)
- **After Fusion (dec4)**: Features after text-image fusion (768 channels, 4×4×4 spatial resolution)

Both feature maps are upsampled to the original image size (128×128×128) and overlaid on the MRI scan.

## Scripts

### 1. `visualize_attention_heatmaps.py`
Main script to visualize a single sample.

### 2. `batch_visualize_heatmaps.py`
Batch processing script to visualize multiple samples at once.

## Usage

### Single Sample Visualization

```bash
# Basic usage (visualizes sample 0 with default settings)
cd /Disk1/afrouz/Projects/TextBraTS
python utils/visualize_attention_heatmaps.py

# Visualize a specific sample
python utils/visualize_attention_heatmaps.py --sample_idx 5

# Specify custom paths
python utils/visualize_attention_heatmaps.py \
    --model_path ./runs/TextBraTS_conda/model.pt \
    --data_dir ./data/TextBraTSData \
    --json_path ./Train.json \
    --sample_idx 0 \
    --output_dir ./visualizations/attention_heatmaps

# Multi-slice visualization (shows 5 slices)
python utils/visualize_attention_heatmaps.py \
    --sample_idx 0 \
    --multi_slice \
    --n_slices 5

# Use different channel reduction methods
python utils/visualize_attention_heatmaps.py \
    --sample_idx 0 \
    --reduction_method mean    # Options: mean, max, attention
```

### Batch Visualization

```bash
# Visualize first 10 training samples
cd /Disk1/afrouz/Projects/TextBraTS
python utils/batch_visualize_heatmaps.py --num_samples 10

# Multi-slice batch visualization
python utils/batch_visualize_heatmaps.py \
    --num_samples 10 \
    --multi_slice \
    --n_slices 5 \
    --output_dir ./visualizations/batch_multislice
```

## Arguments

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `./runs/TextBraTS_conda/model.pt` | Path to pretrained model |
| `--data_dir` | `./data/TextBraTSData` | Dataset directory |
| `--json_path` | `./Train.json` | Dataset JSON file |
| `--output_dir` | `./visualizations/attention_heatmaps` | Output directory |
| `--sample_idx` | `0` | Sample index to visualize |
| `--reduction_method` | `attention` | Channel reduction method |
| `--device` | `cuda` | Device to use (cuda/cpu) |

### Single Sample Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--slice_idx` | `None` | Specific slice to visualize (auto if None) |
| `--multi_slice` | `False` | Generate multi-slice visualization |
| `--n_slices` | `5` | Number of slices for multi-slice mode |

### Batch Processing Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_samples` | `10` | Number of samples to process |

## Channel Reduction Methods

To convert multi-channel features to a single heatmap, three methods are available:

1. **`attention`** (recommended): Computes L2 norm across channels
   - Highlights regions with high activation magnitude
   - Best represents attention-like patterns

2. **`mean`**: Average activation across channels
   - Smoother, more uniform heatmaps
   - Good for general feature magnitude

3. **`max`**: Maximum activation across channels
   - Highlights most activated features
   - Can be sparse

## Output

### Single Sample Output
- **Filename**: `{sample_id}_attention_heatmap.png`
- **Layout**: 1 row × 3 columns
  - Column 1: MRI (FLAIR) + Ground Truth Tumor
  - Column 2: Before Fusion Heatmap (enc3)
  - Column 3: After Fusion Heatmap (dec4)

### Multi-Slice Output
- **Filename**: `{sample_id}_attention_multislice.png`
- **Layout**: N rows × 3 columns (N = number of slices)
- Shows multiple axial slices through the tumor region

## Example Workflow

```bash
# Navigate to project directory
cd /Disk1/afrouz/Projects/TextBraTS

# 1. Visualize a single sample (quick check)
python utils/visualize_attention_heatmaps.py --sample_idx 0

# 2. Generate multi-slice visualization for better understanding
python utils/visualize_attention_heatmaps.py \
    --sample_idx 0 \
    --multi_slice \
    --n_slices 7

# 3. Batch process multiple samples for paper figures
python utils/batch_visualize_heatmaps.py \
    --num_samples 20 \
    --output_dir ./visualizations/paper_figures

# 4. Compare different reduction methods
for method in mean max attention; do
    python utils/visualize_attention_heatmaps.py \
        --sample_idx 0 \
        --reduction_method $method \
        --output_dir ./visualizations/method_comparison_$method
done
```

## Technical Details

### Feature Extraction

The script extracts intermediate features from the model:

- **enc3** (before fusion):
  - Location: After `encoder4` block, before text-image fusion
  - Shape: (1, 192, 16, 16, 16)
  - Represents image features before incorporating text information

- **dec4** (after fusion):
  - Location: After `encoder10` block (bottleneck), after fusion
  - Shape: (1, 768, 4, 4, 4)
  - Represents fused image-text features

### Upsampling

Features are upsampled to 128×128×128 using trilinear interpolation:

```python
F.interpolate(features, size=(128, 128, 128), mode='trilinear', align_corners=True)
```

### Normalization

Heatmaps are normalized to [0, 1] range for consistent visualization:

```python
normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
```

## Colormap

The visualization uses a custom "hot" colormap:
- **Black** → **Purple** → **Red** → **Orange** → **Yellow** → **White**
- Higher values (white/yellow) indicate stronger attention/activation
- Lower values (black/purple) indicate weaker attention

## Dependencies

Required packages (already in your environment):
- `torch`
- `numpy`
- `nibabel`
- `matplotlib`
- `monai`

## Troubleshooting

### Issue: CUDA out of memory
```bash
# Use CPU instead
python utils/visualize_attention_heatmaps.py --device cpu
```

### Issue: Sample not found
```bash
# Check available samples
python -c "import json; print(len(json.load(open('./Train.json'))['training']))"
```

### Issue: Model fails to load
```bash
# Check model path
ls -lh ./runs/TextBraTS_conda/model.pt

# Verify checkpoint structure
python -c "import torch; ckpt = torch.load('./runs/TextBraTS_conda/model.pt', map_location='cpu'); print(ckpt.keys())"
```

## Citation

If you use these visualizations in your work, please cite the TextBraTS paper.

## Notes

- The script automatically selects a tumor-containing slice if `--slice_idx` is not specified
- Multi-slice mode shows slices evenly distributed through the tumor region
- Heatmaps are overlaid with 60% opacity for better visibility of underlying anatomy
- Ground truth tumor segmentation is shown in red with 40% opacity
