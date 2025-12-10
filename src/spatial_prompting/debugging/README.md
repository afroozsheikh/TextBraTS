# Spatial Prompting Debug Tool

Simplified debugging tool for TextBraTS spatial prompting validation.

## What It Does

Generates a **single comprehensive PDF report** containing:

1. **Visualizations** - All samples with:
   - FLAIR reference image
   - Ground truth masks (TC, WT, ET)
   - Atlas masks (TC, WT, ET)
   - Overlays (Red=GT, Green=Atlas)

2. **Overlap Metrics** - For each sample and overall:
   - TC (Tumor Core) overlap percentage
   - WT (Whole Tumor) overlap percentage
   - ET (Enhancing Tumor) overlap percentage

3. **Learned Alpha** - Network's trust parameter for atlas masks

4. **Overall Summary** - Statistics and recommendation

## Quick Start

```bash
# Basic usage with defaults
python spatial_debug.py

# Process 20 random samples
python spatial_debug.py --num-samples 20

# Custom paths
python spatial_debug.py \
    --data-dir /path/to/Merged \
    --atlas-masks-dir /path/to/masks \
    --checkpoint-path /path/to/model.pt \
    --output my_report.pdf
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `/Disk1/afrouz/Data/Merged` | BraTS data directory |
| `--atlas-masks-dir` | `/Disk1/afrouz/Data/TextBraTS_atlas_masks_fixed` | Atlas masks directory |
| `--checkpoint-path` | `/Disk1/afrouz/Projects/TextBraTS/runs/TextBraTS_conda_spatial_prompting/model.pt` | Model checkpoint |
| `--output` | `outputs/spatial_debug_report.pdf` | Output PDF path |
| `--num-samples` | All | Number of samples to process |
| `--seed` | 42 | Random seed for sampling |

## Output

### Terminal Output

Prints progress and metrics:
```
================================================================================
SPATIAL PROMPTING COMPREHENSIVE DEBUG REPORT
================================================================================

1. Loading learned alpha from checkpoint...
   Learned Alpha: 0.753 - Moderately trusts masks

2. Loading samples...
   Found 20 samples to process

3. Processing samples and generating visualizations...
   [1/20] BraTS20_Training_001: TC= 67.3%, WT= 71.2%, ET= 65.8%
   [2/20] BraTS20_Training_002: TC= 72.1%, WT= 75.4%, ET= 68.9%
   ...

================================================================================
OVERALL SUMMARY
================================================================================

Samples Processed: 20

Overlap Statistics:
  TC: Mean= 68.45% ✓ GOOD
  WT: Mean= 72.31% ✓ GOOD
  ET: Mean= 66.78% ✓ GOOD

Learned Alpha: 0.753 - Moderately trusts masks

✓ Report saved to: outputs/spatial_debug_report.pdf
  Total pages: 22
```

### PDF Report Structure

1. **Cover Page** - Processing status and learned alpha
2. **Sample Pages** - One page per sample with:
   - 3×3 grid of visualizations
   - Per-sample overlap metrics in title
3. **Summary Page** - Overall statistics and recommendation

## Interpreting Results

### Overlap Metrics

| Overlap % | Quality | Symbol |
|-----------|---------|--------|
| >80% | Excellent | ✓✓ EXCELLENT |
| 60-80% | Good | ✓ GOOD |
| 40-60% | Moderate | ~ MODERATE |
| <40% | Poor | ⚠️ POOR |

**What it means**: Percentage of ground truth tumor voxels covered by atlas masks. Higher is better.

### Learned Alpha

| Alpha | Interpretation |
|-------|----------------|
| >0.8 | Strongly trusts masks |
| 0.5-0.8 | Moderately trusts masks |
| 0.2-0.5 | Weakly trusts masks |
| <0.2 | Ignoring masks |

**What it means**: How much the network relies on atlas masks during training.

### Recommendation

- **✓✓ PROCEED WITH TRAINING** - Overlap >60%, masks align well
- **~ CAUTIOUSLY PROCEED** - Overlap 40-60%, masks are okay
- **⚠️ DO NOT USE** - Overlap <40%, masks don't align well

## Examples

```bash
# Generate report for all samples
python spatial_debug.py

# Sample 30 random cases
python spatial_debug.py --num-samples 30

# Use different atlas masks
python spatial_debug.py \
    --atlas-masks-dir /Disk1/afrouz/Data/TextBraTS_atlas_masks_aal_v3_padded

# Custom output location
python spatial_debug.py \
    --output /path/to/my_analysis.pdf \
    --num-samples 50 \
    --seed 123
```

## What Changed?

This replaces the previous 7-script debugging system with a single unified tool:

**Before**:
- 7 separate scripts
- Hard-coded paths
- Manual editing required
- Scattered outputs

**After**:
- Single script
- CLI arguments
- One comprehensive PDF report
- Terminal summary

## Help

```bash
python spatial_debug.py --help
```
