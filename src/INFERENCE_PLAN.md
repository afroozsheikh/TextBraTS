# Zero-Training Brain Tumor Localization with Foundation Models

## Overview

This document outlines a complete plan to use **pretrained foundation models** for brain tumor localization from 3D NIfTI MRI scans with text prompts - **NO TRAINING REQUIRED**.

### Architecture

```
3D NIfTI Volume (T1, T1ce, T2, FLAIR) + Text Query
                    ↓
        ┌───────────────────────────┐
        │  Pretrained Grounding DINO │  (Zero-shot text-guided detection)
        │  Process 2D slices         │
        └───────────┬───────────────┘
                    ↓
            2D Bounding Boxes
                    ↓
        ┌───────────────────────────┐
        │  Aggregate to 3D Boxes     │  (Cluster detections across slices)
        └───────────┬───────────────┘
                    ↓
        ┌───────────────────────────┐
        │  Pretrained SAM-Med3D      │  (Refine boxes → masks)
        └───────────┬───────────────┘
                    ↓
    Approximate Lesion Locations + Masks
```

---

## Foundation Models Used

### 1. **Grounding DINO** (Text → Bounding Boxes)
- **Repository**: https://github.com/IDEA-Research/GroundingDINO
- **Paper**: ECCV 2024 - "Grounding DINO: Marrying DINO with Grounded Pre-Training"
- **Capabilities**:
  - Zero-shot object detection with text prompts
  - No training on medical images needed
  - Can detect with queries like "brain tumor", "enhancing lesion", etc.
- **Adaptation Strategy**:
  - Process 3D MRI as 2D slices (axial/coronal/sagittal)
  - Aggregate detections across slices → 3D bounding boxes
  - Convert 4-channel MRI to RGB (use FLAIR or combine modalities)

### 2. **SAM-Med3D** (Boxes → Refined Masks)
- **Repository**: https://github.com/uni-medical/SAM-Med3D
- **Paper**: ECCV 2024 BIC Oral - "Towards General-Purpose Segmentation Models"
- **Capabilities**:
  - Native 3D segmentation (not slice-based)
  - Trained on 143K 3D medical masks across 245 anatomical categories
  - Supports box, point, and mask prompts
  - **Works directly with NIfTI format**
- **Usage**:
  - Takes 3D bounding boxes from Grounding DINO as prompts
  - Outputs refined 3D segmentation masks

---

## Why This Approach?

### ✅ Advantages

1. **No Training Required**
   - Use pretrained models out-of-the-box
   - No need for labeled BraTS data
   - Immediate deployment

2. **Leverages State-of-the-Art Foundation Models**
   - Grounding DINO: Best zero-shot detector (ECCV 2024)
   - SAM-Med3D: Largest 3D medical segmentation model
   - Both models recently published (2024)

3. **Flexible Text Queries**
   - "brain tumor"
   - "enhancing glioblastoma"
   - "tumor in right frontal lobe"
   - "necrotic core"
   - Any natural language description

4. **Native 3D Support**
   - SAM-Med3D processes full 3D volumes
   - Preserves spatial context
   - No slice-by-slice artifacts

5. **Fast Inference**
   - ~5-10 seconds per case
   - Can process entire BraTS dataset quickly
   - GPU: NVIDIA A6000 or similar

---

## Implementation Plan

### Phase 1: Environment Setup (30 minutes)

```bash
# Create conda environment
conda create -n brain_inference python=3.10
conda activate brain_inference

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install groundingdino-py
git clone https://github.com/uni-medical/SAM-Med3D.git
cd SAM-Med3D && pip install -e . && cd ..
pip install nibabel SimpleITK monai matplotlib scipy pandas

# Download pretrained weights
mkdir pretrained_models
# Grounding DINO
wget -O pretrained_models/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# SAM-Med3D (download from GitHub releases)
# https://github.com/uni-medical/SAM-Med3D/releases
# Place sam_med3d_turbo.pth in pretrained_models/
```

---

### Phase 2: Core Pipeline Implementation

#### **Main Components**

**1. NIfTI Loader**
- Load 4-channel BraTS volumes (T1, T1ce, T2, FLAIR)
- Handle both single 4D files and separate modality files
- Extract voxel spacing from NIfTI header

**2. 2D Slice Processor**
- Extract key slices (axial: every 4th slice, or 16 evenly-spaced)
- Convert 4-channel MRI → 3-channel RGB
  - Option A: Replicate FLAIR (best tumor contrast)
  - Option B: Combine modalities (R=T1ce, G=FLAIR, B=T2)
- Normalize to [0, 255] for Grounding DINO

**3. Grounding DINO Inference**
```python
from groundingdino.util.inference import load_model, predict

# Load model
model = load_model(config_path, checkpoint_path)

# Detect per slice
boxes, scores, labels = predict(
    model=model,
    image=rgb_slice,
    caption="brain tumor",  # Text query
    box_threshold=0.25,
    text_threshold=0.20
)
# Returns: boxes in [x1, y1, x2, y2] format
```

**4. 3D Box Aggregation**
- Group detections by spatial proximity
- Cluster boxes across consecutive slices
- Merge overlapping detections (3D NMS)
- Output: 3D boxes [x1, y1, z1, x2, y2, z2]

**5. SAM-Med3D Refinement**
```python
from segment_anything_3d import sam_model_registry, SamPredictor

# Load SAM
sam_model = sam_model_registry["vit_b"](checkpoint="sam_med3d_turbo.pth")
predictor = SamPredictor(sam_model)

# Set image
predictor.set_image(mri_volume)  # (C, H, W, D)

# Predict with box prompts
mask, score, _ = predictor.predict(
    box=box_3d,  # [x1, y1, z1, x2, y2, z2]
    multimask_output=False
)
# Returns: 3D segmentation mask
```

**6. Output Generation**
- Bounding boxes (approximate lesion locations)
- Confidence scores
- Optional: Refined segmentation masks
- Natural language summary:
  ```
  ✅ Detected 2 lesion(s):
    Lesion 1:
      Location: (125.3, 98.7, 78.2) voxels
      Size: 42.0 × 38.0 × 24.0 voxels
      Volume: ~172,584 mm³ (~172.6 cm³)
      Confidence: 78.3%
  ```

---

### Phase 3: Usage Examples

#### **Single Case Inference**

```python
from brain_tumor_localizer import BrainTumorLocalizer

# Initialize (loads pretrained models)
localizer = BrainTumorLocalizer(
    grounding_dino_ckpt="pretrained_models/groundingdino_swint_ogc.pth",
    sam_med3d_ckpt="pretrained_models/sam_med3d_turbo.pth"
)

# Run inference
result = localizer.localize(
    nifti_path="/Disk1/afrouz/Data/Merged/BraTS20_Training_001",
    text_query="brain tumor",
    use_sam=True  # Set False for faster inference (boxes only)
)

# Output
print(result['summary'])
# boxes_3d: List of [x1, y1, z1, x2, y2, z2]
# scores: Confidence scores
# masks: 3D segmentation masks (if use_sam=True)
```

#### **Batch Processing**

```python
# Process all BraTS cases
import os
from pathlib import Path

data_dir = "/Disk1/afrouz/Data/Merged"
cases = sorted([d for d in Path(data_dir).iterdir() if 'BraTS' in d.name])

for case_dir in cases:
    result = localizer.localize(
        nifti_path=str(case_dir),
        text_query="brain tumor",
        use_sam=False  # Faster
    )
    # Save results...
```

#### **Custom Text Queries**

```python
# Different queries
queries = [
    "brain tumor",
    "enhancing glioblastoma",
    "tumor in right hemisphere",
    "necrotic core",
    "edema region"
]

for query in queries:
    result = localizer.localize(nifti_path, text_query=query)
```

---

### Phase 4: Evaluation

#### **Metrics**

1. **Detection Performance**
   - **mAP@IoU[0.5]**: Mean Average Precision at 50% IoU
   - **Localization Accuracy**: % boxes within 10mm of ground truth center
   - Target: mAP ~0.70-0.80 (based on related work)

2. **Segmentation Performance** (with SAM)
   - **Dice Score**: TC, WT, ET
   - **Hausdorff Distance**: 95th percentile
   - Target: Dice ~0.68-0.82 (SAM on BraTS baseline)

3. **Box-Mask Consistency**
   - IoU between predicted boxes and masks
   - Ensures spatial coherence

#### **Expected Performance**

Based on related publications:

| Metric | Expected | Reference |
|--------|----------|-----------|
| Detection mAP@0.5 | 0.70-0.80 | Point-supervised MedSAM (2024) |
| Localization Accuracy | 85-90% | MedSAM box prompts |
| Segmentation Dice | 0.68-0.82 | SAM evaluation on BraTS (Nature 2024) |
| Inference Time | 5-10 sec | SAM-Med3D benchmarks |

#### **Comparison Script**

```python
# Compare predicted boxes with ground truth
from evaluation import compute_metrics

# Load ground truth segmentation
gt_seg = nib.load("BraTS_001_seg.nii.gz").get_fdata()
gt_boxes = generate_boxes_from_segmentation(gt_seg)

# Compute metrics
metrics = compute_metrics(
    pred_boxes=result['boxes_3d'],
    gt_boxes=gt_boxes,
    pred_masks=result['masks'],
    gt_seg=gt_seg
)

print(f"mAP@0.5: {metrics['map_50']:.3f}")
print(f"Dice TC: {metrics['dice_tc']:.3f}")
print(f"Dice WT: {metrics['dice_wt']:.3f}")
print(f"Dice ET: {metrics['dice_et']:.3f}")
```

---

## File Structure

```
TextBraTS/
├── inference/
│   ├── brain_tumor_localizer.py      # Main inference class
│   ├── nifti_utils.py                 # NIfTI loading utilities
│   ├── visualization.py               # Visualization tools
│   ├── evaluation.py                  # Metrics computation
│   ├── run_single_case.py             # Example: single inference
│   └── run_batch.py                   # Example: batch processing
│
├── pretrained_models/
│   ├── groundingdino_swint_ogc.pth    # Grounding DINO weights
│   └── sam_med3d_turbo.pth            # SAM-Med3D weights
│
├── SAM-Med3D/                         # Cloned repository
│
├── GroundingDINO/                     # Cloned repository (for config)
│
└── outputs/
    └── BraTS20_Training_001/
        ├── boxes_3d.npy               # Predicted boxes
        ├── mask_0.nii.gz              # Refined masks
        ├── visualization.png          # 3-view visualization
        └── summary.txt                # Text summary
```

---

## Workflow Summary

### Quick Start (10 minutes)

```bash
# 1. Setup
bash inference/setup.sh

# 2. Download SAM-Med3D weights manually
# From: https://github.com/uni-medical/SAM-Med3D/releases
# Place in: pretrained_models/sam_med3d_turbo.pth

# 3. Run inference on a single case
python inference/run_single_case.py \
    --input /Disk1/afrouz/Data/Merged/BraTS20_Training_001 \
    --query "brain tumor" \
    --output outputs/

# 4. Check results
cat outputs/BraTS20_Training_001/summary.txt
```

### Batch Processing (2-3 hours for ~300 BraTS cases)

```bash
# Process entire dataset
python inference/run_batch.py \
    --data-dir /Disk1/afrouz/Data/Merged \
    --output-dir outputs/batch_results

# Results saved as CSV
cat outputs/batch_results/summary.csv
```

---

## Alternative Approaches (If Needed)

### Option 1: Slice-Based Only (No SAM)
- **Pros**: Faster (~2 sec/case), simpler
- **Cons**: Only bounding boxes, no refined masks
- **Use case**: Quick screening

### Option 2: Fine-tune on BraTS (Optional)
- **Pros**: Better accuracy on brain tumors
- **Cons**: Requires training (2-3 days)
- **Method**: LoRA fine-tuning (train <2% of parameters)
- **Expected improvement**: +5-10% mAP

### Option 3: Ensemble Multiple Queries
```python
queries = ["brain tumor", "glioblastoma", "enhancing lesion"]
all_boxes = []
for query in queries:
    result = localizer.localize(nifti_path, query)
    all_boxes.extend(result['boxes_3d'])
# Merge overlapping boxes
final_boxes = non_max_suppression_3d(all_boxes)
```

---

## Current TextBraTS vs Foundation Models

| Feature | TextBraTS (Current) | Grounding DINO + SAM-Med3D |
|---------|---------------------|----------------------------|
| **Training** | Required (200 epochs, 2-3 days) | ❌ None (pretrained only) |
| **Input** | 4-ch MRI + BioBERT features | 4-ch MRI + text string |
| **Output** | Dense segmentation | Boxes + Optional masks |
| **Inference Time** | 2-3 sec | 5-10 sec |
| **Flexibility** | Fixed classes (TC/WT/ET) | Any text query |
| **Generalization** | BraTS-specific | Broader (143K training masks) |
| **Implementation** | 2-3 weeks | 1-2 days |

---

## Next Steps

### Immediate (Today)
1. ✅ Review this plan
2. ⬜ Decide: Quick prototype (slice-based) or Full pipeline (SAM-Med3D)
3. ⬜ Set up environment (30 min)

### Short-term (This Week)
4. ⬜ Implement core pipeline (2-3 days)
5. ⬜ Test on 5-10 BraTS cases
6. ⬜ Validate outputs qualitatively

### Medium-term (Next Week)
7. ⬜ Batch process full BraTS dataset
8. ⬜ Compute quantitative metrics
9. ⬜ Compare with your TextBraTS results
10. ⬜ Create visualizations for paper/presentation

---

## Resources

### Papers
- **Grounding DINO**: https://arxiv.org/abs/2303.05499
- **SAM-Med3D**: https://arxiv.org/abs/2310.15161
- **Grounding DINO-US-SAM**: https://arxiv.org/abs/2506.23903 (LoRA fine-tuning)
- **SAM on Brain Tumors**: https://www.nature.com/articles/s41598-024-72342-x

### Code Repositories
- **Grounding DINO**: https://github.com/IDEA-Research/GroundingDINO
- **SAM-Med3D**: https://github.com/uni-medical/SAM-Med3D
- **MedSAM Brain**: https://github.com/vpulab/med-sam-brain

### Pretrained Weights
- **Grounding DINO**: https://github.com/IDEA-Research/GroundingDINO/releases
- **SAM-Med3D**: https://github.com/uni-medical/SAM-Med3D/releases

---

## Questions?

**Q: Do I need to modify my existing TextBraTS code?**
A: No, this is a standalone pipeline. You can keep your current implementation.

**Q: Can I use this with non-BraTS data?**
A: Yes! Works with any 3D brain MRI in NIfTI format.

**Q: What if Grounding DINO doesn't detect tumors well?**
A: Try different text queries, adjust thresholds, or process more slices. Can also fine-tune with LoRA if needed.

**Q: How accurate is this compared to supervised methods?**
A: Zero-shot: ~70-80% of supervised performance. With LoRA fine-tuning: ~90-95%.

**Q: Can I run this on CPU?**
A: Technically yes, but very slow (10-20x slower). GPU strongly recommended.

---

## Summary

This plan provides a **complete zero-training solution** for brain tumor localization using state-of-the-art foundation models. You can:

1. ✅ Process 3D NIfTI volumes directly
2. ✅ Use flexible text prompts
3. ✅ Get approximate lesion locations (boxes)
4. ✅ Optionally refine with SAM-Med3D (masks)
5. ✅ Deploy immediately (no training)

**Estimated time to working prototype: 1-2 days**

Ready to implement when you are! 🚀
