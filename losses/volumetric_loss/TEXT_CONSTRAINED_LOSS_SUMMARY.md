# Text-Constrained Loss Functions for BraTS Segmentation

## Overview
Implementation of text-guided loss functions to align predicted BraTS segmentation masks with anatomical locations and volumetric constraints extracted from radiology reports.

---

## Current State Analysis

### Existing Assets ✅
1. **Working baseline**: TextSwinUNETR model with DiceLoss training
2. **Rich structured data**: 369 samples with detailed JSON extractions including:
   - Anatomical regions (Frontal/Parietal/Temporal/Occipital Lobes, sides)
   - Volumetric extents (HIGH/MODERATE/LOW) per pathology and region
   - Multiple pathologies: Lesion, Edema, Necrosis, Mass_Effect
3. **Training infrastructure**: Sample ID extraction, batch processing ready
4. **Text features**: BioBERT embeddings (768-dim) loaded per sample

### Problem Statement
User wants to implement loss functions that enforce:
1. **Spatial constraints**: Predicted masks should appear in text-specified anatomical regions
2. **Volumetric constraints**: Predicted volumes should match text-specified extents (HIGH/MODERATE/LOW)

### Key Challenge: Ambiguities to Resolve

#### 1. **Scope Decision**
Three possible implementation levels:
- **Level 1 (Volume-Only)**: Simple volumetric constraint loss using overall burden scores
- **Level 2 (Volume + Soft Location)**: Volume constraints + region-weighted penalties
- **Level 3 (Volume + Atlas Location)**: Full atlas registration with hard spatial constraints

**Required Clarification**: Which level to implement? Level 1 is fastest, Level 3 matches paper most closely but requires atlas registration infrastructure.

#### 2. **BraTS Channel Mapping Ambiguity**
Output: `(B, 3, 128, 128, 128)` for [TC, WT, ET]
JSON has: Lesion, Edema, Necrosis, Mass_Effect

**Mapping questions**:
- Does "Lesion" refer to Whole Tumor (WT), Tumor Core (TC), or Enhancing Tumor (ET)?
- Should Edema be computed as `WT - TC` or does it map to a specific channel?
- How does Necrosis relate to the channels (typically part of TC)?
- Should Mass_Effect be used (it's structural, not directly segmentable)?

**Required Clarification**: Explicit mapping between JSON pathology types and BraTS channels.

#### 3. **Multi-Region Handling Strategy**
JSON shows pathologies spanning multiple regions with different extents:
```json
"Edema": [
  {"Region": "Parietal", "Volumetric_Extent": "HIGH"},
  {"Region": "Frontal", "Volumetric_Extent": "LOW"}
]
```

**Options**:
- Use aggregate `Overall_Burden.Edema_Extent` (simple but loses granularity)
- Weight regions by extent (requires heuristic weighting)
- Apply per-region constraints (requires atlas to know which voxels belong to each region)

**Required Clarification**: How to aggregate multi-region volumetric information?

#### 4. **Volume Threshold Calibration**
HIGH/MODERATE/LOW are qualitative. Need numeric thresholds.

**Options**:
- Data-driven: Analyze ground truth masks to compute percentile-based thresholds per pathology
- Literature-based: Use standard ranges (e.g., LOW<5%, MODERATE=5-15%, HIGH>15% of brain volume)
- Manual: User provides thresholds

**Required Clarification**: Preferred threshold calibration method?

#### 5. **Atlas Registration Infrastructure**
For spatial location constraints, need:
- Brain atlas (Harvard-Oxford, AAL3, or other)
- Registration to 128×128×128 space
- Region label lookup per voxel

**Required Clarification**:
- Should atlas registration be implemented or use a simpler proxy?
- If yes, which atlas? (Harvard-Oxford has ~50 regions, AAL3 has ~170)

---

## Proposed Implementation Approaches

### Approach A: Pragmatic Volume-Only Loss (Recommended Starting Point)

**What it does**:
- Enforces volumetric constraints using `Overall_Burden` aggregates
- No atlas registration required
- Fast to implement and validate

**Components**:
1. **VolumetricConstraintLoss**: ReLU-based margin loss
   - Maps HIGH/MODERATE/LOW to voxel count ranges
   - Penalizes violations: `loss = ReLU(V_pred - V_max) + ReLU(V_min - V_pred)`
   - Applied per pathology type (Lesion, Edema, Necrosis)

2. **Loss wrapper**: Combines Dice + Volumetric losses
   ```python
   L_total = L_dice + λ_vol * L_volumetric
   ```

**Pros**: Simple, no external dependencies, interpretable
**Cons**: Ignores spatial information, coarse-grained

---

### Approach B: Volume + Location with Atlas Registration (Comprehensive)

**What it does**:
- Volumetric constraints (as in Approach A)
- PLUS spatial location constraints using brain atlas

**Additional Components**:
1. **Atlas preprocessing**:
   - Load Harvard-Oxford or AAL3 atlas
   - Register/resample to 128×128×128 space
   - Create region masks for anatomical labels

2. **LocationAlignmentLoss**: Containment penalty
   - For each pathology, get text-specified regions (e.g., "Right Frontal Lobe")
   - Create allowed region mask from atlas
   - Penalize predictions outside allowed regions:
   ```python
   L_loc = sum(P * (1 - A_region)) / (sum(P) + ε)
   ```
   - P = predicted probability, A_region = atlas mask for allowed region

3. **Multi-region volumetric constraints**:
   - Use atlas to decompose predictions into per-region volumes
   - Apply separate volumetric constraints per region

**Pros**: Matches paper methodology, comprehensive spatial+volumetric control
**Cons**: Requires atlas, more complex, needs registration validation

---

### Approach C: Hybrid with Coarse Spatial Priors (Middle Ground)

**What it does**:
- Volumetric constraints using overall burden
- Coarse spatial guidance using hemisphere/lobe centroids (no full atlas)

**Components**:
1. VolumetricConstraintLoss (same as Approach A)
2. **CoarseSpatialLoss**: Simple hemisphere penalty
   - Extract predicted mask centroid
   - Check if centroid matches text-specified hemisphere (Left/Right)
   - Penalty if mismatch (e.g., text says "Right" but centroid is in left hemisphere)

**Pros**: Some spatial awareness without full atlas complexity
**Cons**: Very coarse, may not help much

---

## Technical Implementation Details

### 1. Data Pipeline Integration
**Files to modify/create**:
- `losses/volumetric_constraint_loss.py` - New volumetric loss class
- `losses/location_alignment_loss.py` - New spatial loss class (if Approach B)
- `losses/text_constrained_loss_wrapper.py` - Combined loss wrapper
- `losses/volume_calibration.py` - Threshold calibration utilities
- `losses/atlas_utils.py` - Atlas loading and registration (if Approach B)

**Integration points**:
- Modify `main.py` lines 252-266 to instantiate new loss
- Use existing `trainer_with_volume_loss.py` (already extracts sample IDs)
- Load JSON constraints at initialization, lookup per sample_id during training

### 2. Loss Function Signature
```python
class TextConstrainedLoss(nn.Module):
    def __init__(self,
                 dice_loss,
                 volumetric_json_path,
                 volume_weight=0.1,
                 location_weight=0.1,  # if Approach B
                 volume_thresholds=None,
                 atlas_masks=None):  # if Approach B
        ...

    def forward(self, logits, targets, sample_ids=None):
        """
        logits: (B, 3, H, W, D) - raw model outputs
        targets: (B, 3, H, W, D) - ground truth masks
        sample_ids: List[str] - e.g., ["BraTS20_Training_001", ...]
        """
        # Compute Dice loss
        loss_dice = self.dice_loss(logits, targets)

        if sample_ids is None:
            return loss_dice

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Compute volumetric constraint loss
        loss_vol = self.compute_volumetric_loss(probs, sample_ids)

        # Compute location loss (if enabled)
        loss_loc = self.compute_location_loss(probs, sample_ids) if self.use_location else 0

        return loss_dice + self.volume_weight * loss_vol + self.location_weight * loss_loc
```

### 3. Volume Threshold Mapping
Example structure:
```python
VOLUME_THRESHOLDS = {
    'Lesion': {
        'LOW': (0, 5000),      # voxels
        'MODERATE': (5000, 15000),
        'HIGH': (15000, float('inf'))
    },
    'Edema': {
        'LOW': (0, 8000),
        'MODERATE': (8000, 25000),
        'HIGH': (25000, float('inf'))
    },
    'Necrosis': {
        'LOW': (0, 3000),
        'MODERATE': (3000, 10000),
        'HIGH': (10000, float('inf'))
    }
}
```

These would be computed from dataset statistics or provided manually.

### 4. JSON Lookup Structure
```python
class VolumetricConstraintDatabase:
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)

    def get_constraints(self, sample_id):
        """
        Returns dict with volumetric constraints for this sample
        """
        sample = self.data.get(sample_id)
        if not sample:
            return None

        return {
            'lesion_extent': sample['Overall_Burden']['Lesion_Extent'],
            'edema_extent': sample['Overall_Burden']['Edema_Extent'],
            'regions': self.extract_regions(sample['Pathologies']),
            # ... more as needed
        }
```

---

## Key Resources & Papers

### Primary Reference
**"Learning Segmentation from Radiology Reports"** (arXiv:2507.05582)
- Text-guided segmentation methodology
- Constrained CNN approaches

### Additional Key Papers

1. **"Constrained-CNN losses for weakly supervised segmentation"** (Kervadec et al., MIA 2019)
   - arXiv:1805.04620
   - Definitive paper on differentiable constraint losses
   - Size constraints, topology constraints
   - Directly applicable volumetric loss formulations

2. **"On the inclusion of prior knowledge in deep learning"** (Mosinska et al.)
   - Size and shape priors in medical segmentation
   - Biological constraint enforcement

3. **Brain Atlases**:
   - **Harvard-Oxford Cortical/Subcortical Atlas**: ~50 regions, widely used
   - **AAL3 (Automated Anatomical Labeling)**: ~170 regions, more granular
   - Available via FSL, ANTsPy, or NiLearn

### Software Libraries

1. **MONAI** (already in use): `monai.losses` for base losses
2. **NiBabel**: Loading/manipulating NIfTI atlas files
3. **ANTsPy** or **NiLearn**: Atlas registration if needed
4. **SimpleITK**: Alternative for registration

---

## Critical Open Questions

Before implementation can proceed, need clarity on:

1. **Scope**: Volume-only (Approach A) vs Full atlas-based (Approach B)?
2. **Channel mapping**: How do Lesion/Edema/Necrosis map to TC/WT/ET channels?
3. **Multi-region aggregation**: Use Overall_Burden or per-region constraints?
4. **Thresholds**: Data-driven calibration or manual specification?
5. **Priority**: Fix existing broken volume loss first, or build new system from scratch?

---

## Recommended Next Steps

### Option 1: Quick Validation Path (Recommended)
1. Start with **Approach A** (Volume-only)
2. Use `Overall_Burden` aggregates to avoid multi-region complexity
3. Implement threshold calibration script to analyze ground truth masks
4. Create simple wrapper combining Dice + Volumetric loss
5. Train and validate if volumetric constraints improve alignment
6. **Then** expand to spatial constraints if needed

**Timeline**: Can have working implementation in 1-2 development sessions

### Option 2: Comprehensive Path
1. Implement **Approach B** with full atlas registration
2. Handle multi-region constraints properly
3. More complex but complete solution

**Timeline**: 3-4 development sessions due to atlas integration complexity

---

## Files to Create/Modify

### New Files
- `/Disk1/afrouz/Projects/TextBraTS/losses/volumetric_constraint_loss.py`
- `/Disk1/afrouz/Projects/TextBraTS/losses/text_constrained_loss_wrapper.py`
- `/Disk1/afrouz/Projects/TextBraTS/losses/compute_volume_thresholds.py` (calibration script)
- `/Disk1/afrouz/Projects/TextBraTS/losses/atlas_utils.py` (if Approach B)
- `/Disk1/afrouz/Projects/TextBraTS/losses/location_alignment_loss.py` (if Approach B)

### Modified Files
- `/Disk1/afrouz/Projects/TextBraTS/main.py` (lines 252-266: loss initialization)
- Potentially update training args to add new hyperparameters

### Existing Files to Use
- `/Disk1/afrouz/Projects/TextBraTS/trainer_with_volume_loss.py` (already extracts sample IDs)
- `/Disk1/afrouz/Projects/TextBraTS/losses/volumetric_extractions.json` (369 samples with structured data)

---

## Summary

This project has **excellent structured data** ready to use. The main decisions needed are:
1. How ambitious to be with spatial constraints (atlas vs no atlas)
2. How to map pathology types to BraTS channels
3. How to handle multi-region annotations

**Recommended**: Start with simple volumetric constraints (Approach A), validate the concept works, then incrementally add spatial constraints if needed. This minimizes risk and allows for iterative validation.
