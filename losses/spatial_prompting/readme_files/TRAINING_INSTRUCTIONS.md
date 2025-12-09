# Spatial Prompting Training Instructions

## Quick Start

### Option 1: Training WITH Spatial Prompting (Recommended to Try)

```bash
cd /Disk1/afrouz/Projects/TextBraTS
./train_with_spatial_prompting.sh
```

### Option 2: Baseline Training WITHOUT Spatial Prompting (For Comparison)

```bash
cd /Disk1/afrouz/Projects/TextBraTS
./train_baseline.sh
```

## What You're Using

### Atlas Configuration (Best Performance)
- **Atlas**: AAL (Automated Anatomical Labeling)
- **Version**: v3 expanded with deep structures
- **Processing**: Interpolated (nearest-neighbor)
- **Directory**: `/Disk1/afrouz/Data/TextBraTS_atlas_masks_aal_v3`
- **Mapping**: `region_mapping_aal_v3_expanded.json`

### Performance Metrics
- **Ground Truth Overlap**: 43.6% (moderate, usable)
  - TC (Tumor Core): 43.9%
  - WT (Whole Tumor): 42.8%
  - ET (Enhancing Tumor): 44.3%
- **Brain Coverage**: 6-7% (good specificity)
- **Status**: ✓ Cautiously proceed

## What to Expect

### During Training

1. **Learned Alpha Parameter**
   - Initial value: 0.7
   - Expected final: 0.4-0.6 (moderate trust)
   - Monitor in TensorBoard: `spatial_prompt_alpha`

   **Interpretation:**
   - α > 0.5: Network trusts masks (good!)
   - α ~ 0.3-0.5: Partial trust (reasonable)
   - α < 0.3: Network ignoring masks (masks not helpful)

2. **Performance Improvements**
   - Expected Dice improvement: 1-3%
   - Reduced false positives in unreported regions
   - Better anatomical plausibility

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir=./runs

# Watch for:
# 1. spatial_prompt_alpha (should stabilize around 0.4-0.6)
# 2. Dice scores (should improve by 1-3%)
# 3. Loss curves (should converge normally)
```

## Configuration Details

### Training Parameters
- **Batch size**: 2 (fits on 24GB GPU)
- **Max epochs**: 200
- **Learning rate**: 1e-4
- **Feature size**: 48
- **ROI size**: 128×128×128
- **Validation frequency**: Every epoch

### Spatial Prompting Mechanism

The network uses **learnable soft gating**:

```python
# Learnable parameter (trained end-to-end)
alpha = self.spatial_prompt_alpha  # Range: [0, 1]

# Effective mask blends atlas guidance with unrestricted predictions
effective_mask = atlas_mask * alpha + (1 - alpha)

# Apply to network output
logits = logits * effective_mask
```

**Benefits:**
- Not too harsh (allows predictions everywhere)
- Network learns optimal trust level
- Robust to imperfect atlas masks

## Experimental Comparison Results

We tested multiple approaches. Here's what we found:

| Atlas | Method | GT Overlap | Recommendation |
|-------|--------|-----------|----------------|
| **AAL v3** | **Interpolation** | **43.6%** | **✅ BEST - Use this** |
| AAL v3 | Padding | 37.0% | Worse than interpolated |
| Harvard-Oxford v2 | Padding | 19.2% | Poor - Don't use |

**Key Finding**: Interpolation actually performed better than padding for AAL atlas!
- Interpolated version has 2.3x more labeled voxels (20.5% vs 8.8%)
- Better coverage leads to better overlap with actual tumors

## Files and Directories

### Atlas Files
```
/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/
├── brain_atlas_aal_resampled.nii.gz      # Atlas (128³, interpolated)
├── brain_atlas_aal_padded.nii.gz         # Atlas (padded version)
├── atlas_labels_aal.json                 # Label definitions
└── atlas_visualization_aal.png           # Visualization

/Disk1/afrouz/Data/TextBraTS_atlas_masks_aal_v3/  # 369 sample masks (USE THIS)
├── BraTS20_Training_001_atlas_mask.nii.gz
├── BraTS20_Training_002_atlas_mask.nii.gz
└── ...

/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/
├── region_mapping_aal_v3_expanded.json   # Region mapping (v3)
├── volumetric_extractions.json           # LLM-extracted regions
└── debugging/
    ├── debug_aal_v3_results.txt          # Debug output (interpolated)
    ├── debug_aal_v3_padded_results.txt   # Debug output (padded)
    └── atlas_vs_gt_visualization_v3.pdf  # Visual comparison
```

### Training Scripts
```
/Disk1/afrouz/Projects/TextBraTS/
├── train_with_spatial_prompting.sh       # Main training script
├── train_baseline.sh                     # Baseline (no spatial prompting)
└── main.py                               # Core training code
```

## Troubleshooting

### If Alpha Stays Low (~0.2-0.3)
- Masks aren't helping much
- Still okay - network is learning to ignore them
- Compare with baseline to see if any improvement

### If Training Crashes
- Reduce batch size to 1: `--batch_size 1`
- Check GPU memory: `nvidia-smi`
- Enable gradient checkpointing if available

### If No Improvement Over Baseline
- This is okay - 43.6% overlap is moderate
- Check that masks are being loaded correctly
- Verify atlas_masks_dir path is correct

## Next Steps After Training

1. **Compare Results**
   - Run both spatial prompting and baseline
   - Compare Dice scores (TC, WT, ET)
   - Check learned alpha value

2. **Analyze Predictions**
   - Visualize predictions with/without spatial prompting
   - Check if false positives reduced in unreported regions
   - Verify anatomical plausibility improved

3. **Potential Improvements** (if results are promising)
   - Try different alpha initialization values
   - Experiment with per-channel alpha (separate for TC/WT/ET)
   - Add spatial loss in addition to soft gating
   - Improve LLM extraction quality

## Important Notes

- **Don't expect miracles**: 43.6% overlap is moderate, not excellent
- **Soft gating is key**: Network can learn to partially ignore imperfect masks
- **Monitor alpha**: It tells you how much the network trusts the masks
- **Compare with baseline**: Only way to know if it's actually helping

## Contact & References

- Session summary: `SESSION_SUMMARY_2025-12-08.md`
- Full README: `SPATIAL_PROMPTING_README.md`
- Debug results: `debugging/debug_aal_v3_results.txt`

---

**Last Updated**: 2025-12-08
