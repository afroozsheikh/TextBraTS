# Spatial Loss Approach for TextBraTS

## Overview

This document outlines the approach for implementing **spatial loss** (also called spatial prompting or anatomical constraint loss) for TextBraTS brain tumor segmentation. The goal is to use anatomical location information extracted from radiology reports to guide and constrain the segmentation model's predictions.

---

## Motivation

Radiology reports contain rich anatomical information about tumor locations (e.g., "Right Frontal Lobe", "Parietal Lobe"). Currently, this spatial information is not being explicitly used during training. By incorporating spatial constraints, we can:

1. **Improve segmentation accuracy** - Guide predictions to anatomically plausible regions
2. **Reduce false positives** - Penalize predictions in regions not mentioned in reports
3. **Better leverage multimodal data** - Align text (reports) with vision (segmentation masks)

---

## Core Concept

### The Spatial Constraint

**Problem**: Current loss functions (e.g., Dice Loss) only measure overlap between prediction and ground truth, but don't consider whether predictions appear in anatomically correct locations mentioned in the radiology report.

**Solution**: Add a spatial constraint loss that penalizes predictions that "leak" outside the anatomical regions mentioned in the text.

### Mathematical Formulation

Given:
- **P**: Predicted probability map (B, 3, H, W, D) - soft probabilities after sigmoid
- **A**: Atlas-based region mask (B, 3, H, W, D) - binary mask indicating allowed regions
- **Pathology types**: Lesion (→ WT), Edema (→ WT-TC), Necrosis (→ TC-ET)

The spatial loss penalizes predictions outside allowed regions:

```
L_spatial = Σ (P × (1 - A)) / (Σ P + ε)
```

Where:
- `P × (1 - A)` captures predictions in "forbidden zones" (outside atlas regions)
- Division by `Σ P` normalizes by total prediction volume
- `ε` prevents division by zero

---

## Implementation Pipeline

### Step 1: Brain Atlas Preprocessing ✅ COMPLETED

**Script**: `generate_brain_atlas_masks.py`

**What it does**:
1. Downloads Harvard-Oxford brain atlas (48 cortical + 21 subcortical regions)
2. Resamples atlas to 128×128×128 (BraTS target shape)
3. Saves:
   - `brain_atlas_harvard-oxford_resampled.nii.gz` - 3D label map
   - `atlas_labels_harvard-oxford.json` - Label ID → Region name mapping
   - `atlas_stats_harvard-oxford.json` - Atlas statistics
   - `atlas_visualization_harvard-oxford.png` - Visual slices

**Output**: A spatial reference where each voxel has a label indicating its anatomical region.

### Step 2: Region Mapping (Text → Atlas Labels)

**Next step**: Create a mapping system that converts:
- Text-based regions from JSON (e.g., "Right Frontal Lobe")
- → Atlas label IDs (e.g., [1, 2, 3] for right frontal regions)

**Mapping strategy**:

```python
region_mapping = {
    'Frontal Lobe': {
        'Right': [1, 2, 3, 4],      # Right frontal atlas labels
        'Left': [5, 6, 7, 8],       # Left frontal atlas labels
        'Bilateral': [1, 2, 3, 4, 5, 6, 7, 8]
    },
    'Parietal Lobe': {
        'Right': [9, 10, 11],
        'Left': [12, 13, 14],
        'Bilateral': [9, 10, 11, 12, 13, 14]
    },
    # ... more regions
}
```

**Challenges**:
- Handle region name variations ("Frontal Lobe" vs "Frontal Lobes" vs "Frontal Region")
- Map junction regions ("Junction of Frontal and Parietal Lobes")
- Handle generic terms ("Brain", "Cerebral Hemisphere") → use whole brain mask
- Deal with lesion-specific regions ("Adjacent to Lesion") → may need whole brain

### Step 3: Per-Sample Atlas Mask Generation

**Next script needed**: `create_sample_atlas_masks.py`

For each training sample:
1. Load `volumetric_extractions.json` for that sample
2. Extract anatomical regions per pathology type:
   - Lesion regions → Channel 1 (WT)
   - Edema regions → Channel 1 (WT)
   - Necrosis regions → Channel 0 (TC)
3. Use region mapping + atlas to create 3D binary masks
4. Save as `{sample_id}_atlas_mask.npy` with shape (3, 128, 128, 128)

**Example**:
```json
// BraTS20_Training_001 JSON
"Lesion": [
  {"Region": "Frontal Lobe", "Side": "Right"},
  {"Region": "Parietal Lobe", "Side": "Right"}
]
```

→ Creates mask where:
- Channel 1 (WT): All voxels in right frontal + right parietal = 1, rest = 0
- Channel 0, 2: Similar for other pathologies

### Step 4: Spatial Loss Implementation

**Script needed**: `losses/spatial_constraint_loss.py`

```python
class SpatialConstraintLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred_probs, atlas_masks):
        """
        Args:
            pred_probs: (B, 3, H, W, D) - Predicted probabilities [0, 1]
            atlas_masks: (B, 3, H, W, D) - Binary allowed region masks

        Returns:
            loss: Scalar penalty for predictions outside allowed regions
        """
        # Forbidden zone = complement of allowed regions
        forbidden_zone = 1.0 - atlas_masks

        # Measure prediction "leakage" into forbidden zones
        leakage = pred_probs * forbidden_zone

        # Normalize by total prediction volume
        total_pred = torch.sum(pred_probs, dim=(2, 3, 4)) + 1e-6
        total_leakage = torch.sum(leakage, dim=(2, 3, 4))

        loss = torch.mean(total_leakage / total_pred)

        return self.weight * loss
```

### Step 5: Training Integration

**Modify**: `trainer.py` or training loop

```python
# Initialize losses
dice_loss = DiceLoss()
spatial_loss = SpatialConstraintLoss(weight=0.1)

# In training loop
for batch in dataloader:
    images = batch['image']
    targets = batch['label']
    atlas_masks = batch['atlas_mask']  # NEW: Load atlas masks
    sample_ids = batch['sample_id']

    # Forward pass
    logits = model(images, text_features)
    probs = torch.sigmoid(logits)

    # Compute losses
    loss_dice = dice_loss(logits, targets)
    loss_spatial = spatial_loss(probs, atlas_masks)

    # Combined loss
    total_loss = loss_dice + loss_spatial

    # Backward pass
    total_loss.backward()
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     SPATIAL LOSS PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

1. PREPROCESSING (One-time)
   ┌──────────────────┐
   │ Harvard-Oxford   │
   │ Brain Atlas      │──→ Download & Resample to 128³
   │ (nilearn)        │
   └──────────────────┘
            │
            ├─→ brain_atlas_resampled.nii.gz (3D label map)
            └─→ atlas_labels.json (ID → Region name)

2. SAMPLE-SPECIFIC MASK GENERATION (Training prep)
   ┌──────────────────────────┐
   │ volumetric_extractions   │
   │        .json             │
   └──────────────────────────┘
            │
            ├─→ Extract regions per sample
            │   - Lesion: "Right Frontal Lobe"
            │   - Edema: "Parietal Lobe"
            │
            ├─→ Map text → Atlas label IDs
            │   "Right Frontal" → [1, 2, 3, 4]
            │
            └─→ Create binary masks per channel
                ┌─────────────────────────────┐
                │ {sample_id}_atlas_mask.npy │
                │ Shape: (3, 128, 128, 128)  │
                └─────────────────────────────┘

3. TRAINING (Runtime)
   ┌──────────────┐       ┌──────────────┐
   │ MRI Images   │       │ Ground Truth │
   │ (4 channels) │       │ Segmentation │
   └──────────────┘       └──────────────┘
         │                       │
         ├───────────┬───────────┤
         │           │           │
         ▼           ▼           ▼
   ┌─────────────────────────────────┐
   │      TextSwinUNETR Model        │
   └─────────────────────────────────┘
                │
                ▼
         Predicted Probs (P)
         Shape: (B, 3, 128, 128, 128)
                │
                ├──────────┬──────────┐
                │          │          │
                ▼          ▼          ▼
           Dice Loss   Spatial Loss  Volume Loss
                │          │          │
                │    ┌─────┴─────┐    │
                │    │ Atlas     │    │
                │    │ Masks (A) │    │
                │    └───────────┘    │
                │                     │
                └──────────┬──────────┘
                           │
                    Total Loss = α·Dice + β·Spatial + γ·Volume
```

---

## Pathology → Channel Mapping

As defined in [PATHOLOGY_TO_BRATS_MAPPING.md](PATHOLOGY_TO_BRATS_MAPPING.md):

```
JSON Pathology  →  Model Channel  →  Region Source
─────────────────────────────────────────────────────
Lesion          →  Channel 1 (WT) →  Use Lesion regions from JSON
Edema           →  Channel 1 (WT) →  Use Edema regions from JSON
Necrosis        →  Channel 0 (TC) →  Use Necrosis regions from JSON
Mass_Effect     →  Not used       →  Structural effect, not segmentable
```

**Important**:
- Both Lesion and Edema regions should constrain WT (Channel 1)
- Take **union** of Lesion + Edema regions for Channel 1 atlas mask
- Necrosis regions constrain TC (Channel 0)

---

## Handling Edge Cases

### 1. Generic/Vague Regions
**Problem**: JSON contains regions like "Brain", "Cerebral Hemisphere", "Lesion Region"

**Solution**: For these generic terms, use **whole brain mask** (all atlas regions) rather than specific lobes. This means no spatial constraint (allow predictions anywhere).

### 2. Junction Regions
**Problem**: "Junction of Frontal and Parietal Lobes"

**Solution**: Take **union** of both regions' atlas labels.

### 3. Multiple Regions per Pathology
**Problem**: Lesion appears in both "Frontal Lobe" and "Temporal Lobe"

**Solution**: Create atlas mask as **union** of all mentioned regions.

### 4. Missing JSON Data
**Problem**: Some samples may not have detailed region information

**Solution**:
- Option A: Use whole brain mask (no spatial constraint)
- Option B: Skip spatial loss for that sample (use Dice only)

### 5. Laterality (Left/Right/Bilateral)
**Problem**: "Side" field in JSON specifies Left/Right/Bilateral

**Solution**:
- Right → Only include right hemisphere atlas labels
- Left → Only include left hemisphere atlas labels
- Bilateral → Include both hemispheres
- Midline → Include midline structures (ventricles, corpus callosum)

---

## Loss Weight Tuning

The spatial loss should be weighted carefully to not overwhelm the Dice loss:

```python
Total_Loss = λ_dice · L_dice + λ_spatial · L_spatial + λ_volume · L_volume
```

**Recommended starting values**:
- `λ_dice = 1.0` (primary loss)
- `λ_spatial = 0.1` (gentle spatial guidance)
- `λ_volume = 0.05` (volumetric constraints)

**Tuning strategy**:
1. Start with spatial loss weight = 0.05
2. Monitor metrics:
   - Dice scores (should not decrease)
   - Spatial leakage (% of predictions outside allowed regions)
3. Gradually increase if spatial alignment improves without hurting Dice
4. Typical range: 0.01 to 0.3

---

## Expected Benefits

### 1. Improved Anatomical Plausibility
- Predictions will be spatially coherent with text descriptions
- Reduced false positives in anatomically implausible locations

### 2. Better Use of Multimodal Data
- Currently text features are only used via cross-attention
- Spatial loss explicitly enforces text-vision alignment

### 3. Potential Performance Gains
- Papers show 2-5% Dice improvement with spatial constraints
- Most gains in cases where initial predictions are spatially inconsistent

### 4. Interpretability
- Can visualize where model predictions violate anatomical constraints
- Helps identify model failures vs. annotation errors

---

## Alternative Approaches Considered

### Approach A: Hard Masking (Not Recommended)
Apply atlas mask directly to predictions: `P_constrained = P × A`

**Pros**: Simple, guaranteed spatial constraint
**Cons**:
- Non-differentiable (no gradient flow)
- Too rigid, can hurt performance if atlas/text is imprecise
- Can't learn from mistakes

### Approach B: Soft Spatial Loss (Recommended) ✓
Current approach - penalize but don't prevent predictions outside regions

**Pros**:
- Differentiable, allows gradient flow
- Flexible, model can override if needed
- Gracefully handles imprecise atlas alignment

### Approach C: Weighted Dice Loss
Weight Dice loss by atlas masks to focus on correct regions

**Pros**: Single loss, simpler
**Cons**:
- Doesn't explicitly penalize out-of-region predictions
- Less interpretable

---

## Implementation Checklist

### Phase 1: Foundation ✅
- [x] Download and preprocess brain atlas
- [x] Resample atlas to 128×128×128
- [x] Save atlas label mappings
- [x] Document pathology → channel mapping

### Phase 2: Region Mapping 🔄 IN PROGRESS
- [ ] Create comprehensive text → atlas label mapping
- [ ] Handle region name variations
- [ ] Handle laterality (Left/Right/Bilateral)
- [ ] Handle junction regions
- [ ] Handle generic/vague terms

### Phase 3: Sample Mask Generation 📋 TODO
- [ ] Write script to generate per-sample atlas masks
- [ ] Process all samples in volumetric_extractions.json
- [ ] Save masks as numpy arrays
- [ ] Validate mask quality (spot checks)

### Phase 4: Loss Implementation 📋 TODO
- [ ] Implement SpatialConstraintLoss class
- [ ] Add unit tests
- [ ] Integrate into training pipeline
- [ ] Add atlas mask loading to dataloader

### Phase 5: Training & Evaluation 📋 TODO
- [ ] Train with spatial loss (various weights)
- [ ] Compare Dice scores vs. baseline
- [ ] Measure spatial leakage metric
- [ ] Visualize predictions with/without spatial loss
- [ ] Ablation studies

---

## References

### Key Papers
1. **"Constrained-CNN losses for weakly supervised segmentation"** (Kervadec et al., MIA 2019)
   - Definitive paper on differentiable constraint losses
   - Size and spatial constraint formulations

2. **"Learning Segmentation from Radiology Reports"** (arXiv:2507.05582)
   - Text-guided medical segmentation
   - Similar spatial constraint approach

### Datasets
- **BraTS 2020**: Brain tumor segmentation challenge
- **Harvard-Oxford Atlas**: 48 cortical + 21 subcortical regions
- **TextBraTS**: 369 samples with structured radiology reports

### Code Resources
- `nilearn`: Brain atlas downloading and resampling
- `nibabel`: NIfTI file handling
- `MONAI`: Medical imaging losses and metrics

---

## Contact & Questions

For questions or suggestions about this approach, please refer to:
- Main documentation: `README.md`
- Pathology mapping: `PATHOLOGY_TO_BRATS_MAPPING.md`
- Loss implementations: `TEXT_CONSTRAINED_LOSS_SUMMARY.md`

---

**Last Updated**: 2025-12-05
**Status**: Phase 1 Complete, Phase 2 In Progress
