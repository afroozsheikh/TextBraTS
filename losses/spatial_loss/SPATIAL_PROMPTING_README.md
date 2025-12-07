# Spatial Prompting for Medical Image Segmentation

## Overview

This document describes the complete pipeline for implementing **spatial prompting** in text-guided medical image segmentation for the TextBraTS dataset. Spatial prompting uses brain atlas masks derived from radiology reports to guide the segmentation network toward anatomically plausible predictions.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Pipeline Overview](#pipeline-overview)
3. [Step 1: LLM-Based Anatomical Extraction](#step-1-llm-based-anatomical-extraction)
4. [Step 2: Region Mapping Creation](#step-2-region-mapping-creation)
5. [Step 3: Sample-Specific Atlas Mask Generation](#step-3-sample-specific-atlas-mask-generation)
6. [Step 4: Network Integration](#step-4-network-integration)
7. [Training with Spatial Prompting](#training-with-spatial-prompting)
8. [Implementation Details](#implementation-details)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

### Motivation

Traditional text-guided segmentation methods use text embeddings (e.g., BioBERT) to condition the network, but these embeddings lack explicit spatial information. Radiology reports contain valuable anatomical location information (e.g., "lesion in right frontal lobe"), which can be leveraged to create **spatial prompts** that guide the network.

### What is Spatial Prompting?

Spatial prompting uses anatomical atlas masks to:
- **Guide predictions** to anatomically plausible regions mentioned in the radiology report
- **Reduce false positives** outside reported locations
- **Improve text-image alignment** by bridging textual descriptions and 3D anatomy

### Key Concept

For each sample:
1. Extract anatomical regions from the radiology report (e.g., "right frontal lobe")
2. Map these regions to Harvard-Oxford brain atlas labels
3. Generate a 3D binary mask indicating allowed regions for each pathology type
4. Use these masks as spatial attention in the segmentation network

---

## Pipeline Overview

```
Radiology Reports (Text)
    ↓
[1] LLM Extraction (mri_report_extractor.py)
    ↓
Structured Anatomical Data (JSON)
    ↓
[2] Region Mapping (create_region_mapping.py)
    ↓
Atlas Label Mappings
    ↓
[3] Atlas Mask Generation (generate_sample_atlas_masks.py)
    ↓
Per-Sample Atlas Masks (3×128×128×128)
    ↓
[4] Network Integration (data_utils.py, textswin_unetr.py, trainer.py)
    ↓
Spatially-Guided Predictions
```

---

## Step 1: LLM-Based Anatomical Extraction

### Purpose
Extract structured anatomical and volumetric information from free-text radiology reports.

### Script
`losses/spatial_loss/mri_report_extractor.py`

### What It Does

The script uses Large Language Models (LLMs) to parse radiology reports and extract:
- **Anatomical regions**: Which brain structures are affected (Frontal Lobe, Temporal Lobe, etc.)
- **Laterality**: Left, Right, Bilateral, or Midline
- **Pathology types**: Lesion, Edema, Necrosis, Mass Effect
- **Volumetric descriptors**: HIGH, MODERATE, LOW extent

### Input
Free-text radiology reports from TextBraTS dataset:
```
"The brain MRI reveals a large lesion in the right frontal and temporal lobes,
with the frontal lobe showing a mixture of high and low signal intensities..."
```

### Output
Structured JSON with anatomical annotations:
```json
{
  "Report_ID": "BraTS20_Training_001",
  "Pathologies": {
    "Lesion": [
      {
        "Region": "Frontal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "HIGH",
        "Descriptors": ["large", "mixture"],
        "Signal_Characteristics": ["mixture high/low signal"],
        "Spatial_Pattern": ["involving frontal lobe"]
      }
    ],
    "Edema": [...],
    "Necrosis": [...],
    "Mass_Effect": [...]
  }
}
```

### Usage

#### Process all samples:
```bash
cd /Disk1/afrouz/Projects/TextBraTS/losses/spatial_loss

python mri_report_extractor.py \
    --data_dir /Disk1/afrouz/Data/Merged \
    --output volumetric_extractions.json \
    --provider anthropic \
    --api_key YOUR_API_KEY
```

#### Process a single sample (for testing):
```bash
python mri_report_extractor.py \
    --data_dir /Disk1/afrouz/Data/Merged \
    --sample BraTS20_Training_001 \
    --provider anthropic
```

#### Supported LLM Providers:
- **Anthropic Claude** (recommended): `--provider anthropic`
- **OpenAI GPT-4**: `--provider openai`
- **Google Gemini**: `--provider gemini`

### Key Features

1. **Anatomical Precision**: Identifies specific brain regions and laterality
2. **Volumetric Quantification**: Categorizes extent as HIGH/MODERATE/LOW
3. **Multi-Pathology Support**: Handles Lesion, Edema, Necrosis, Mass Effect
4. **Resume Support**: Can resume interrupted processing
5. **Error Handling**: Saves progress after each sample

### Output File
`volumetric_extractions.json` - Contains structured data for all samples

---

## Step 2: Region Mapping Creation

### Purpose
Create a mapping from text-based anatomical region descriptions to Harvard-Oxford brain atlas label IDs.

### Script
`losses/spatial_loss/create_region_mapping.py`

### What It Does

Maps anatomical region names (e.g., "Right Frontal Lobe") to specific label IDs in the Harvard-Oxford brain atlas.

### Harvard-Oxford Atlas Structure

- **Labels 1-48**: Cortical regions (bilateral)
  - Frontal: 1, 3-7, 25-28, 33, 41
  - Parietal: 17-21, 31, 43
  - Temporal: 8-16, 34-35, 37-39, 44-46
  - Occipital: 22-24, 32, 36, 40, 47-48
  - Insula: 2
  - Cingulate: 29-30

- **Labels 49-69**: Subcortical structures (lateralized)
  - Left hemisphere: 49-59
  - Right hemisphere: 60-69
  - Includes: Basal Ganglia, Thalamus, Hippocampus, Amygdala, Brainstem

### Usage

```bash
cd /Disk1/afrouz/Projects/TextBraTS/losses/spatial_loss

python create_region_mapping.py
```

### Input
- `atlas_labels_harvard-oxford.json`: Harvard-Oxford atlas label definitions

### Output
`region_mapping.json`: Mapping structure
```json
{
  "Frontal Lobe": {
    "Left": [1, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 33, 41],
    "Right": [1, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 33, 41],
    "Bilateral": [1, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 33, 41],
    "Unspecified": [1, 3, 4, 5, 6, 7, 25, 26, 27, 28, 29, 33, 41]
  },
  "Basal Ganglia": {
    "Left": [53, 54, 55, 59],
    "Right": [64, 65, 66, 69],
    "Bilateral": [53, 54, 55, 59, 64, 65, 66, 69],
    "Unspecified": [53, 54, 55, 59, 64, 65, 66, 69]
  }
}
```

### Key Features

1. **Medical Accuracy**: Separates cortical lobes from deep white matter and subcortical nuclei
2. **Laterality Support**: Handles Left/Right/Bilateral/Unspecified
3. **Junction Handling**: Maps compound regions like "Temporo-Parietal Junction"
4. **Fallback Logic**: Whole brain mask for vague descriptions

---

## Step 3: Sample-Specific Atlas Mask Generation

### Purpose
Generate per-sample 3D binary atlas masks indicating anatomically plausible regions for each pathology type.

### Script
`losses/spatial_loss/generate_sample_atlas_masks.py`

### What It Does

For each sample in the dataset:
1. Reads the LLM-extracted anatomical annotations
2. Maps region names to atlas labels using the region mapping
3. Creates a 3-channel 3D mask (3×128×128×128)
4. Saves the mask as a NIfTI file

### Channel Mapping

Based on BraTS label conventions:
- **Channel 0 (TC - Tumor Core)**: Necrosis regions
- **Channel 1 (WT - Whole Tumor)**: Lesion + Edema regions (union)
- **Channel 2 (ET - Enhancing Tumor)**: Lesion regions (proxy for enhancing)

### Usage

```bash
cd /Disk1/afrouz/Projects/TextBraTS/losses/spatial_loss

python generate_sample_atlas_masks.py
```

### Configuration

Edit paths in `main()` function:
```python
atlas_path = "/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/brain_atlas_harvard-oxford_resampled.nii.gz"
region_mapping_path = "/Disk1/afrouz/Projects/TextBraTS/losses/spatial_loss/region_mapping.json"
volumetric_extractions_path = "/Disk1/afrouz/Projects/TextBraTS/losses/volumetric_extractions.json"
output_dir = "/Disk1/afrouz/Data/TextBraTS_atlas_masks"
```

### Input Files

1. **Brain Atlas**: `brain_atlas_harvard-oxford_resampled.nii.gz` (128×128×128)
2. **Region Mapping**: `region_mapping.json`
3. **Volumetric Extractions**: `volumetric_extractions.json`

### Output

For each sample (e.g., `BraTS20_Training_001`):
- **Atlas Mask**: `BraTS20_Training_001_atlas_mask.nii.gz` (3×128×128×128)
- **Visualization**: `BraTS20_Training_001_atlas_mask_viz.png` (optional)

### Example Output

For a report mentioning "right frontal lobe lesion with edema":
```
Channel 0 (TC): Empty → Whole brain fallback
Channel 1 (WT): Right frontal lobe mask (Lesion + Edema)
Channel 2 (ET): Right frontal lobe mask (Lesion)
```

### Fallback Mechanism

If a channel has no regions (empty mask), it defaults to whole brain:
```python
whole_brain_mask = self.create_mask_from_labels(list(range(1, 70)))
if tc_mask.sum() == 0:
    tc_mask = whole_brain_mask.copy()
```

### Visualization

Generates axial slice visualizations showing all 3 channels:
```
[TC Channel]  [WT Channel]  [ET Channel]
```

---

## Step 4: Network Integration

### Purpose
Integrate atlas masks as spatial prompts into the text-guided segmentation network.

### Implementation Strategy: **Learnable Soft Gating**

Instead of hard masking (which completely zeros predictions outside atlas regions), we use **learnable soft gating** that allows the network to control how much to trust the atlas masks.

#### The Soft Gating Formula

```python
# Learnable parameter (initialized at 0.7, trained end-to-end)
alpha = self.spatial_prompt_alpha  # Range: [0, 1]

# Effective mask blends atlas guidance with unrestricted predictions
effective_mask = atlas_mask * alpha + (1 - alpha)

# Apply to network output
logits = logits * effective_mask
```

#### How It Works

- **α = 1.0**: Hard masking - full trust in atlas (only predict in atlas regions)
- **α = 0.7**: Default - strong guidance from atlas, but allows some predictions outside
- **α = 0.5**: Balanced - equal weight to atlas and non-atlas regions
- **α = 0.0**: No masking - ignore atlas completely (baseline behavior)

The network learns the optimal α during training via backpropagation.

#### Why Soft Gating?

**Hard Masking Problems:**
- Completely zeros predictions outside atlas regions → too harsh
- Can't recover from imperfect atlas masks
- No flexibility for unexpected tumor locations

**Soft Gating Benefits:**
- Allows predictions everywhere, but weighted by atlas guidance
- Network learns how much to trust the spatial priors
- Gracefully handles imperfect or missing atlas information
- More robust to atlas generation errors

### Modified Files

#### 4.1 Data Loading: `utils/data_utils.py`

**Added `LoadAtlasMaskd` transform class:**
```python
class LoadAtlasMaskd(MapTransform):
    def __init__(self, keys, atlas_masks_dir):
        super().__init__(keys)
        self.atlas_masks_dir = atlas_masks_dir

    def __call__(self, data):
        # Extract sample ID from image path
        sample_id = os.path.basename(os.path.dirname(image_path))

        # Load atlas mask
        atlas_mask_path = os.path.join(
            self.atlas_masks_dir,
            f"{sample_id}_atlas_mask.nii.gz"
        )

        if os.path.exists(atlas_mask_path):
            atlas_nii = nib.load(atlas_mask_path)
            d[key] = atlas_nii.get_fdata().astype(np.float32)
        else:
            # Fallback to whole brain
            d[key] = np.ones((3, 128, 128, 128), dtype=np.float32)

        return d
```

**Updated `get_loader()` function:**
```python
def get_loader(args):
    # Determine if we should load atlas masks
    load_atlas = hasattr(args, 'spatial_prompting') and args.spatial_prompting

    if load_atlas:
        train_transform_list.append(
            LoadAtlasMaskd(keys=["atlas_mask"], atlas_masks_dir=args.atlas_masks_dir)
        )
        tensor_keys.append("atlas_mask")
```

#### 4.2 Network Architecture: `utils/textswin_unetr.py`

**Added learnable parameter in `__init__()`:**
```python
# Learnable spatial prompting weight (initialized at 0.7)
self.spatial_prompt_alpha = nn.Parameter(torch.tensor(0.7))
```

**Updated `forward()` method with soft gating:**
```python
def forward(self, x_in, text_in, atlas_mask=None):
    """
    Args:
        x_in: Input images (B, 4, H, W, D)
        text_in: Text embeddings (B, text_dim)
        atlas_mask: Spatial prompts from atlas (B, 3, H, W, D) - optional
    """
    # Normal encoder-decoder processing
    hidden_states_out = self.swinViT(x_in, text_in, self.normalize)
    # ... encoder and decoder layers ...
    logits = self.out(out)

    # Apply atlas mask as soft spatial attention (learnable gating)
    if atlas_mask is not None:
        # Soft gating: blend atlas guidance with unrestricted predictions
        alpha = torch.clamp(self.spatial_prompt_alpha, 0.0, 1.0)
        effective_mask = atlas_mask * alpha + (1.0 - alpha)
        logits = logits * effective_mask

    return logits
```

**Why this approach?**
- **Learnable**: Network learns optimal trust level for atlas masks
- **Robust**: Gracefully handles imperfect atlas generation
- **Flexible**: Can range from hard masking (α=1) to no masking (α=0)
- **Non-invasive**: Doesn't change input channels or core architecture
- **Interpretable**: Clear how spatial priors influence predictions

#### 4.3 Training Loop: `trainer.py`

**Updated `train_epoch()`:**
```python
def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    for idx, batch_data in enumerate(loader):
        data = batch_data["image"]
        target = batch_data["label"]
        text = batch_data["text_feature"]
        atlas_mask = batch_data.get("atlas_mask", None)  # Load atlas mask

        # Move to GPU
        data = data.cuda(args.rank)
        target = target.cuda(args.rank)
        text = text.cuda(args.rank)
        if atlas_mask is not None:
            atlas_mask = atlas_mask.cuda(args.rank)

        # Forward pass with atlas mask
        with autocast('cuda', enabled=args.amp):
            if atlas_mask is not None:
                logits = model(data, text, atlas_mask=atlas_mask)
            else:
                logits = model(data, text)
            loss = loss_func(logits, target)
```

**Updated `val_epoch()`:**
Similar changes to handle atlas masks during validation.

#### 4.4 Command-Line Interface: `main.py`

**Added argument:**
```python
parser.add_argument(
    "--spatial_prompting",
    action="store_true",
    help="use atlas masks as spatial prompts for the network"
)
parser.add_argument(
    "--atlas_masks_dir",
    default="/Disk1/afrouz/Data/TextBraTS_atlas_masks",
    type=str,
    help="directory containing per-sample atlas masks"
)
```

---

## Training with Spatial Prompting

### Basic Training Command

```bash
cd /Disk1/afrouz/Projects/TextBraTS

python main.py \
    --data_dir /Disk1/afrouz/Data/Merged \
    --json_list ./Train.json \
    --batch_size 2 \
    --max_epochs 200 \
    --spatial_prompting \
    --atlas_masks_dir /Disk1/afrouz/Data/TextBraTS_atlas_masks \
    --save_checkpoint \
    --logdir spatial_prompting_experiment
```

### Advanced Training Options

#### With Distributed Training (Multi-GPU):
```bash
python main.py \
    --data_dir /Disk1/afrouz/Data/Merged \
    --json_list ./Train.json \
    --batch_size 2 \
    --max_epochs 200 \
    --spatial_prompting \
    --atlas_masks_dir /Disk1/afrouz/Data/TextBraTS_atlas_masks \
    --distributed \
    --world_size 2 \
    --save_checkpoint \
    --logdir spatial_prompting_multi_gpu
```

#### Without Spatial Prompting (Baseline):
```bash
python main.py \
    --data_dir /Disk1/afrouz/Data/Merged \
    --json_list ./Train.json \
    --batch_size 2 \
    --max_epochs 200 \
    --save_checkpoint \
    --logdir baseline_no_prompting
```

#### With Custom Atlas Directory:
```bash
python main.py \
    --spatial_prompting \
    --atlas_masks_dir /path/to/custom/atlas/masks \
    ...
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--spatial_prompting` | False | Enable spatial prompting |
| `--atlas_masks_dir` | `/Disk1/afrouz/Data/TextBraTS_atlas_masks` | Atlas masks directory |
| `--batch_size` | 2 | Training batch size |
| `--max_epochs` | 200 | Number of training epochs |
| `--optim_lr` | 1e-4 | Learning rate |
| `--feature_size` | 48 | Network feature size |
| `--roi_x/y/z` | 128 | Input patch size |
| `--val_every` | 1 | Validation frequency |

### Expected Behavior

#### With Spatial Prompting ON:
- Atlas masks are loaded and passed to the network
- Network output is multiplied by atlas masks
- Predictions are suppressed outside reported regions
- Console output shows: "Loading atlas mask for BraTS20_Training_XXX"

#### With Spatial Prompting OFF:
- No atlas masks are loaded
- Network operates normally without spatial constraints
- All regions are equally considered for prediction

---

## Implementation Details

### Data Flow

```
Sample: BraTS20_Training_001
    ↓
Load Image: (4, 240, 240, 155) → Resize → (4, 128, 128, 128)
Load Label: (3, 240, 240, 155) → Resize → (3, 128, 128, 128)
Load Text: (768,) BioBERT embedding
Load Atlas: BraTS20_Training_001_atlas_mask.nii.gz → (3, 128, 128, 128)
    ↓
Batch: {
    "image": (B, 4, 128, 128, 128),
    "label": (B, 3, 128, 128, 128),
    "text_feature": (B, 768),
    "atlas_mask": (B, 3, 128, 128, 128)  # if spatial_prompting=True
}
    ↓
Network:
    Encoder-Decoder → logits (B, 3, 128, 128, 128)
    if atlas_mask:
        alpha = clamp(learnable_alpha, 0, 1)  # Currently: 0.7 → learned during training
        effective_mask = atlas_mask * alpha + (1 - alpha)
        logits = logits * effective_mask
    ↓
Loss: DiceLoss(logits, label)
```

### Memory Considerations

**Additional Memory per Sample:**
- Atlas mask: 3 × 128 × 128 × 128 × 4 bytes = ~25 MB (float32)
- Batch of 2: ~50 MB additional GPU memory

**Recommendations:**
- Keep batch size at 2 for 24GB GPUs
- Use gradient checkpointing for larger batches
- Monitor GPU memory usage during training

### Computational Overhead

**Additional Operations:**
- Loading atlas masks: ~10-20ms per sample (disk I/O)
- Element-wise multiplication: Negligible (<1ms on GPU)

**Total Overhead:** <5% of training time

---

## Troubleshooting

### Issue 1: Atlas Mask Not Found
```
Warning: Atlas mask not found for BraTS20_Training_001, using whole brain
```

**Solution:**
- Verify atlas masks were generated for all samples
- Check `atlas_masks_dir` path is correct
- Run `generate_sample_atlas_masks.py` if missing

### Issue 2: Shape Mismatch
```
RuntimeError: The size of tensor a (3) must match the size of tensor b (4)
```

**Cause:** Atlas mask has wrong number of channels

**Solution:**
- Regenerate atlas masks ensuring 3 channels (TC, WT, ET)
- Check atlas mask generation script output

### Issue 3: Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size: `--batch_size 1`
- Enable gradient checkpointing: `--use_checkpoint`
- Use mixed precision: Don't use `--noamp`

### Issue 4: All Predictions Zeroed
```
Dice scores: TC=0.0, WT=0.0, ET=0.0
```

**Cause:** Atlas masks are all zeros

**Solution:**
- Check volumetric_extractions.json has valid data
- Verify fallback to whole brain is working
- Inspect a few atlas masks: `nibabel.load(mask_path).get_fdata()`

### Issue 5: No Improvement with Spatial Prompting
```
Results similar to baseline
```

**Possible Causes:**
1. Atlas masks too permissive (whole brain fallback for most samples)
2. Network already learning spatial patterns well
3. Need to tune other hyperparameters

**Solutions:**
- Analyze atlas mask coverage statistics
- Try different LLM prompts for more specific regions
- Combine with spatial loss for stronger constraint

---

## Validation and Analysis

### Monitor Learned Alpha Parameter

During training, you can monitor how much the network learns to trust the atlas masks:

```python
# During training or after loading a checkpoint
model = TextSwinUNETR(...)  # Your model
model.load_state_dict(checkpoint['state_dict'])

# Print current alpha value
alpha = model.spatial_prompt_alpha.item()
print(f"Learned spatial prompting alpha: {alpha:.4f}")

# Interpretation:
# α ≈ 1.0: Network strongly trusts atlas masks
# α ≈ 0.5: Network moderately uses atlas guidance
# α ≈ 0.0: Network learned to ignore atlas masks (may indicate poor quality)
```

**Add to TensorBoard logging:**
```python
# In trainer.py, add to run_training():
if args.rank == 0 and writer is not None:
    # Log alpha parameter every epoch
    alpha_value = model.spatial_prompt_alpha.item() if not args.distributed else \
                  model.module.spatial_prompt_alpha.item()
    writer.add_scalar("spatial_prompt_alpha", alpha_value, epoch)
```

### Check Atlas Mask Quality

```python
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load atlas mask
mask = nib.load("/Disk1/afrouz/Data/TextBraTS_atlas_masks/BraTS20_Training_001_atlas_mask.nii.gz")
mask_data = mask.get_fdata()

# Check coverage
print(f"TC coverage: {(mask_data[0] > 0).sum() / mask_data[0].size * 100:.2f}%")
print(f"WT coverage: {(mask_data[1] > 0).sum() / mask_data[1].size * 100:.2f}%")
print(f"ET coverage: {(mask_data[2] > 0).sum() / mask_data[2].size * 100:.2f}%")

# Visualize middle slice
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, title in enumerate(['TC', 'WT', 'ET']):
    axes[i].imshow(mask_data[i, :, :, 64].T, cmap='hot', origin='lower')
    axes[i].set_title(title)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('atlas_mask_check.png', dpi=150)
```

### Analyze Atlas Coverage Statistics

```python
import json
import nibabel as nib
from pathlib import Path
import numpy as np

atlas_dir = Path("/Disk1/afrouz/Data/TextBraTS_atlas_masks")
mask_files = list(atlas_dir.glob("*_atlas_mask.nii.gz"))

stats = {'TC': [], 'WT': [], 'ET': []}

for mask_file in mask_files:
    mask = nib.load(mask_file).get_fdata()
    stats['TC'].append((mask[0] > 0).sum() / mask[0].size * 100)
    stats['WT'].append((mask[1] > 0).sum() / mask[1].size * 100)
    stats['ET'].append((mask[2] > 0).sum() / mask[2].size * 100)

for channel in ['TC', 'WT', 'ET']:
    print(f"\n{channel} Coverage Statistics:")
    print(f"  Mean: {np.mean(stats[channel]):.2f}%")
    print(f"  Median: {np.median(stats[channel]):.2f}%")
    print(f"  Min: {np.min(stats[channel]):.2f}%")
    print(f"  Max: {np.max(stats[channel]):.2f}%")
```

---

## File Structure

```
TextBraTS/
├── main.py                          # Main training script (updated)
├── trainer.py                       # Training loop (updated)
├── utils/
│   ├── data_utils.py                # Data loading (updated)
│   └── textswin_unetr.py            # Network architecture (updated)
├── losses/
│   ├── volumetric_extractions.json  # LLM extraction output
│   └── spatial_loss/
│       ├── mri_report_extractor.py  # Step 1: LLM extraction
│       ├── create_region_mapping.py # Step 2: Region mapping
│       ├── generate_sample_atlas_masks.py  # Step 3: Mask generation
│       └── region_mapping.json      # Region mapping output
└── SPATIAL_PROMPTING_README.md      # This file

Data/
├── Merged/                          # BraTS dataset
│   └── BraTS20_Training_XXX/
│       ├── *_t1.nii
│       ├── *_t2.nii
│       ├── *_flair.nii
│       ├── *_t1ce.nii
│       ├── *_seg.nii
│       └── *_flair_text.txt         # Radiology reports
├── TextBraTS_atlas_preprocess/
│   ├── brain_atlas_harvard-oxford_resampled.nii.gz  # Atlas (128³)
│   └── atlas_labels_harvard-oxford.json             # Atlas labels
└── TextBraTS_atlas_masks/           # Generated atlas masks
    ├── BraTS20_Training_001_atlas_mask.nii.gz
    ├── BraTS20_Training_002_atlas_mask.nii.gz
    └── ...
```

---

## Citation

If you use this spatial prompting approach in your research, please cite:

```bibtex
@article{textbrats_spatial_prompting_2025,
  title={Spatial Prompting for Text-Guided Medical Image Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Important Notes

### Channel Order Verification

⚠️ **Critical**: Ensure atlas mask channels match BraTS label convention:

**MONAI's `ConvertToMultiChannelBasedOnBratsClassesd`:**
- **Channel 0**: TC (Tumor Core) = Label 1 (Necrosis) OR Label 4 (Enhancing)
- **Channel 1**: WT (Whole Tumor) = Label 1 OR 2 OR 4 (all tumor components)
- **Channel 2**: ET (Enhancing Tumor) = Label 4 only

**Your Atlas Generation Should Match:**
- **Channel 0**: TC from Necrosis + Lesion regions (both contribute to tumor core)
- **Channel 1**: WT from Lesion + Edema regions (whole tumor extent)
- **Channel 2**: ET from Lesion regions (proxy for enhancing tumor)

If you observe poor performance, verify channel alignment between atlas masks and ground truth labels.

### Soft Gating Benefits Recap

1. **No Hard Cutoffs**: Predictions allowed everywhere, weighted by atlas
2. **Learnable Trust**: Network learns optimal α parameter (0 to 1)
3. **Robust to Errors**: Gracefully handles imperfect atlas generation
4. **Interpretable**: Can monitor α to see how much network trusts spatial priors

---

## Future Improvements

1. ~~**Learnable Attention Weights**~~: ✅ Implemented as soft gating with learnable alpha
2. **Multi-Scale Prompting**: Apply atlas masks at multiple decoder levels
3. **Uncertainty-Aware Prompting**: Weight atlas influence based on text confidence
4. **Dynamic Atlas Selection**: Learn which atlas regions to trust per sample
5. **Cross-Attention Fusion**: Fuse atlas masks with image features via attention
6. **Per-Channel Alpha**: Learn different α values for TC, WT, ET channels

---

## Contact

For questions or issues, please contact:
- Email: your.email@university.edu
- GitHub: https://github.com/yourusername/TextBraTS

---

## Acknowledgments

- Harvard-Oxford brain atlas from FSL (FMRIB Software Library)
- BraTS dataset organizers
- MONAI framework for medical imaging
- Anthropic Claude / OpenAI GPT-4 / Google Gemini for LLM extraction

---

**Last Updated:** 2025-12-07
