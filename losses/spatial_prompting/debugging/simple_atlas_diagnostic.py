"""
Simple atlas mask diagnostic (no torch required)
"""

import nibabel as nib
import numpy as np
from pathlib import Path

def check_atlas_coverage(atlas_masks_dir="/Disk1/afrouz/Data/TextBraTS_atlas_masks"):
    """Check how restrictive atlas masks are"""
    atlas_dir = Path(atlas_masks_dir)

    if not atlas_dir.exists():
        print(f"Atlas directory not found: {atlas_masks_dir}")
        return None

    mask_files = list(atlas_dir.glob("*_atlas_mask.nii.gz"))

    if len(mask_files) == 0:
        print(f"No atlas masks found in {atlas_masks_dir}")
        return None

    stats = {'TC': [], 'WT': [], 'ET': []}
    whole_brain_count = {'TC': 0, 'WT': 0, 'ET': 0}

    print(f"Analyzing {len(mask_files)} atlas masks...")

    for mask_file in mask_files:
        try:
            mask_data = nib.load(str(mask_file)).get_fdata()

            # Calculate coverage (percentage of brain that's allowed)
            for i, channel in enumerate(['TC', 'WT', 'ET']):
                coverage = (mask_data[i] > 0).sum() / mask_data[i].size * 100
                stats[channel].append(coverage)

                # Check if it's essentially whole brain (>95% coverage)
                if coverage > 95:
                    whole_brain_count[channel] += 1
        except Exception as e:
            print(f"Error loading {mask_file}: {e}")
            continue

    print("\n" + "="*70)
    print("ATLAS MASK COVERAGE STATISTICS")
    print("="*70)

    for channel in ['TC', 'WT', 'ET']:
        if len(stats[channel]) > 0:
            print(f"\n{channel} Channel:")
            print(f"  Mean coverage: {np.mean(stats[channel]):.2f}%")
            print(f"  Median coverage: {np.median(stats[channel]):.2f}%")
            print(f"  Min coverage: {np.min(stats[channel]):.2f}%")
            print(f"  Max coverage: {np.max(stats[channel]):.2f}%")
            print(f"  Std dev: {np.std(stats[channel]):.2f}%")
            print(f"  Whole brain fallback: {whole_brain_count[channel]}/{len(mask_files)} ({whole_brain_count[channel]/len(mask_files)*100:.1f}%)")

    # Analysis and recommendations
    print("\n" + "="*70)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*70)

    for channel in ['TC', 'WT', 'ET']:
        if len(stats[channel]) > 0:
            mean_coverage = np.mean(stats[channel])
            fallback_rate = whole_brain_count[channel] / len(mask_files) * 100

            if fallback_rate > 70:
                print(f"\n⚠️  {channel}: HIGH FALLBACK RATE ({fallback_rate:.1f}%)")
                print(f"   → Atlas masks are mostly whole brain (not restrictive)")
                print(f"   → Spatial prompting likely NOT adding constraint")

            elif mean_coverage > 80:
                print(f"\n⚠️  {channel}: LARGE COVERAGE ({mean_coverage:.1f}%)")
                print(f"   → Atlas masks cover most of the brain")
                print(f"   → Limited spatial guidance benefit")

            elif mean_coverage < 50:
                print(f"\n✓ {channel}: GOOD COVERAGE ({mean_coverage:.1f}%)")
                print(f"   → Atlas masks are restrictive")
                print(f"   → Should provide meaningful spatial guidance")

            else:
                print(f"\n~ {channel}: MODERATE COVERAGE ({mean_coverage:.1f}%)")
                print(f"   → Atlas masks provide some constraint")

    return stats


def analyze_soft_gating_effect(alpha=0.7):
    """
    Explain the soft gating effect given the observed logit changes
    """
    print("\n" + "="*70)
    print("SOFT GATING ANALYSIS")
    print("="*70)

    print(f"\nCurrent alpha value: {alpha}")
    print(f"\nSoft gating formula: effective_mask = atlas_mask × {alpha} + (1 - {alpha})")
    print(f"\nEffect on logits:")
    print(f"  • Inside atlas regions (mask=1): effective_mask = {alpha} → logits multiplied by {alpha + (1-alpha):.1f}")
    print(f"  • Outside atlas regions (mask=0): effective_mask = {1-alpha} → logits multiplied by {1-alpha:.1f}")

    print(f"\nYour observations:")
    print(f"  • Logits BEFORE mask: -872,026.5")
    print(f"  • Logits AFTER mask:  -261,702.4")
    print(f"  • Reduction: {((872026.5 - 261702.4) / 872026.5 * 100):.1f}%")

    print(f"\nWhat this means:")
    print(f"  ✓ The mask IS actively working (70% reduction in magnitude)")
    print(f"  ✓ Predictions outside atlas regions are being suppressed")

    print(f"\nWhy you might not see improvement:")
    print(f"  1. Atlas masks may be too permissive (see coverage stats above)")
    print(f"  2. Atlas masks may not align well with true tumor locations")
    print(f"  3. The network may need to learn a different alpha value")
    print(f"  4. Suppression of false positives may be offset by suppression of true positives")


def check_sample_atlas_mask(sample_id, atlas_masks_dir="/Disk1/afrouz/Data/TextBraTS_atlas_masks"):
    """Detailed analysis of a single sample's atlas mask"""
    atlas_file = Path(atlas_masks_dir) / f"{sample_id}_atlas_mask.nii.gz"

    if not atlas_file.exists():
        print(f"\nAtlas mask not found for {sample_id}: {atlas_file}")
        return

    print("\n" + "="*70)
    print(f"SAMPLE-SPECIFIC ANALYSIS: {sample_id}")
    print("="*70)

    mask_data = nib.load(str(atlas_file)).get_fdata()

    print(f"\nAtlas mask shape: {mask_data.shape}")
    print(f"Expected shape: (3, 128, 128, 128)")

    if mask_data.shape[0] != 3:
        print(f"⚠️  WARNING: Unexpected number of channels ({mask_data.shape[0]})")

    for i, channel in enumerate(['TC (Tumor Core)', 'WT (Whole Tumor)', 'ET (Enhancing Tumor)']):
        coverage = (mask_data[i] > 0).sum() / mask_data[i].size * 100
        unique_vals = np.unique(mask_data[i])

        print(f"\n{channel}:")
        print(f"  Coverage: {coverage:.2f}%")
        print(f"  Unique values: {unique_vals}")
        print(f"  Min: {mask_data[i].min():.4f}, Max: {mask_data[i].max():.4f}")
        print(f"  Mean: {mask_data[i].mean():.4f}")

        if coverage > 95:
            print(f"  ⚠️  Likely whole brain fallback")
        elif coverage < 30:
            print(f"  ✓ Highly specific/restrictive mask")


if __name__ == "__main__":
    print("="*70)
    print("SPATIAL PROMPTING DIAGNOSTIC TOOL")
    print("="*70)

    # 1. Check atlas mask coverage statistics
    stats = check_atlas_coverage()

    # 2. Analyze soft gating effect
    analyze_soft_gating_effect(alpha=0.7)

    # 3. Check a specific sample
    check_sample_atlas_mask("BraTS20_Training_001")

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)

    print("\n📊 RECOMMENDATIONS:")
    print("1. Check the coverage statistics above")
    print("2. If most masks are whole brain, your LLM extraction may need improvement")
    print("3. Look at volumetric_extractions.json to see if regions are being extracted")
    print("4. The mask IS working (based on your logit sums), but may not be helping")
    print("5. Consider checking Dice scores with/without spatial prompting to see actual benefit")
