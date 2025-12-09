"""
Comprehensive Harvard-Oxford Atlas Debugging Script

Checks:
1. Atlas labels and coverage
2. Region mapping correctness
3. Sample atlas masks (coverage and quality)
4. Ground truth overlap analysis
5. Visualization of masks vs GT
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def check_atlas():
    """Check the Harvard-Oxford atlas file."""
    print("="*80)
    print("1. CHECKING HARVARD-OXFORD ATLAS")
    print("="*80)

    atlas_path = "/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/brain_atlas_harvard-oxford_padded.nii.gz"

    atlas = nib.load(atlas_path)
    atlas_data = atlas.get_fdata()

    print(f"\nAtlas file: {atlas_path}")
    print(f"  Shape: {atlas_data.shape}")
    print(f"  Dtype: {atlas_data.dtype}")

    unique_labels = sorted([int(x) for x in np.unique(atlas_data) if x > 0])
    print(f"  Unique labels: {len(unique_labels)}")
    print(f"  Label range: {min(unique_labels)} - {max(unique_labels)}")

    # Check label distribution
    total_voxels = atlas_data.size
    background_voxels = (atlas_data == 0).sum()
    labeled_voxels = total_voxels - background_voxels

    print(f"\n  Voxel statistics:")
    print(f"    Total voxels: {total_voxels:,}")
    print(f"    Background (0): {background_voxels:,} ({background_voxels/total_voxels*100:.1f}%)")
    print(f"    Labeled: {labeled_voxels:,} ({labeled_voxels/total_voxels*100:.1f}%)")

    # Sample some label counts
    print(f"\n  Sample label coverage (top 10 by voxel count):")
    label_counts = {}
    for label in unique_labels:
        count = (atlas_data == label).sum()
        label_counts[label] = count

    top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for label, count in top_labels:
        pct = count / total_voxels * 100
        print(f"    Label {label}: {count:7,} voxels ({pct:5.2f}%)")

    return unique_labels


def check_region_mapping(unique_labels):
    """Check the region mapping file."""
    print("\n" + "="*80)
    print("2. CHECKING REGION MAPPING")
    print("="*80)

    mapping_path = "/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/region_mapping_harvard-oxford_v2.json"

    with open(mapping_path, 'r') as f:
        region_mapping = json.load(f)

    print(f"\nMapping file: {mapping_path}")
    print(f"  Regions defined: {len(region_mapping)}")

    # Check which labels are used
    all_mapped_labels = set()
    for region, sides in region_mapping.items():
        for side, labels in sides.items():
            all_mapped_labels.update(labels)

    print(f"  Unique labels in mapping: {len(all_mapped_labels)}")

    # Check if mapped labels exist in atlas
    atlas_labels_set = set(unique_labels)
    mapped_but_not_in_atlas = all_mapped_labels - atlas_labels_set
    in_atlas_but_not_mapped = atlas_labels_set - all_mapped_labels

    print(f"\n  Label validation:")
    print(f"    Labels in mapping AND atlas: {len(all_mapped_labels & atlas_labels_set)}")
    print(f"    Labels in mapping but NOT in atlas: {len(mapped_but_not_in_atlas)}")
    if mapped_but_not_in_atlas:
        print(f"      Examples: {sorted(list(mapped_but_not_in_atlas))[:10]}")
    print(f"    Labels in atlas but NOT in mapping: {len(in_atlas_but_not_mapped)}")
    if in_atlas_but_not_mapped:
        print(f"      Examples: {sorted(list(in_atlas_but_not_mapped))[:10]}")

    # Show some example mappings
    print(f"\n  Example mappings:")
    for region in ['Frontal Lobe', 'Parietal Lobe', 'Temporal Lobe', 'Basal Ganglia']:
        if region in region_mapping:
            labels_right = region_mapping[region].get('Right', [])
            print(f"    {region:20s} (Right): {len(labels_right)} labels")
            if labels_right:
                print(f"      First 5: {labels_right[:5]}")

    return region_mapping


def check_sample_masks(num_samples=10):
    """Check generated Harvard-Oxford atlas masks."""
    print("\n" + "="*80)
    print("3. CHECKING SAMPLE ATLAS MASKS")
    print("="*80)

    masks_dir = Path("/Disk1/afrouz/Data/TextBraTS_atlas_masks_harvard-oxford_v2")

    if not masks_dir.exists():
        print(f"\n  ⚠️  Masks directory not found: {masks_dir}")
        print("  Run: python generate_sample_atlas_masks.py")
        return None

    mask_files = list(masks_dir.glob("*_atlas_mask.nii.gz"))

    print(f"\nMasks directory: {masks_dir}")
    print(f"  Total masks found: {len(mask_files)}")

    if len(mask_files) == 0:
        print("  No masks found!")
        return None

    # Sample random masks
    sample_files = random.sample(mask_files, min(num_samples, len(mask_files)))

    stats = {'TC': [], 'WT': [], 'ET': []}

    print(f"\n  Analyzing {len(sample_files)} random samples:")
    for mask_file in sample_files:
        sample_id = mask_file.stem.replace('_atlas_mask', '')
        mask = nib.load(str(mask_file)).get_fdata()

        total_voxels = mask[0].size
        tc_cov = (mask[0] > 0).sum() / total_voxels * 100
        wt_cov = (mask[1] > 0).sum() / total_voxels * 100
        et_cov = (mask[2] > 0).sum() / total_voxels * 100

        stats['TC'].append(tc_cov)
        stats['WT'].append(wt_cov)
        stats['ET'].append(et_cov)

        print(f"    {sample_id}: TC={tc_cov:5.2f}%, WT={wt_cov:5.2f}%, ET={et_cov:5.2f}%")

    # Overall statistics
    print(f"\n  Coverage Statistics (from {len(sample_files)} samples):")
    for channel in ['TC', 'WT', 'ET']:
        if stats[channel]:
            print(f"    {channel}:")
            print(f"      Mean:   {np.mean(stats[channel]):6.2f}%")
            print(f"      Median: {np.median(stats[channel]):6.2f}%")
            print(f"      Min:    {np.min(stats[channel]):6.2f}%")
            print(f"      Max:    {np.max(stats[channel]):6.2f}%")
            print(f"      Std:    {np.std(stats[channel]):6.2f}%")

    return stats


def check_ground_truth_overlap(num_samples=20):
    """Check overlap between atlas masks and ground truth."""
    print("\n" + "="*80)
    print("4. CHECKING GROUND TRUTH OVERLAP")
    print("="*80)

    data_dir = Path("/Disk1/afrouz/Data/Merged")
    masks_dir = Path("/Disk1/afrouz/Data/TextBraTS_atlas_masks_harvard-oxford_v2")

    if not masks_dir.exists():
        print(f"  ⚠️  Masks directory not found")
        return None

    # Get sample directories
    sample_dirs = [d for d in data_dir.iterdir() if d.is_dir() and 'BraTS' in d.name]

    if len(sample_dirs) == 0:
        print(f"  No samples found in {data_dir}")
        return None

    # Random sample
    sample_dirs = random.sample(sample_dirs, min(num_samples, len(sample_dirs)))

    overlaps = {'TC': [], 'WT': [], 'ET': []}

    print(f"\n  Analyzing {len(sample_dirs)} samples...")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name

        # Load ground truth segmentation
        seg_files = list(sample_dir.glob("*_seg.nii*"))
        if len(seg_files) == 0:
            continue

        seg_data = nib.load(seg_files[0]).get_fdata()

        # Convert to 3-channel format (TC, WT, ET)
        gt_tc = ((seg_data == 1) | (seg_data == 4)).astype(float)
        gt_wt = ((seg_data == 1) | (seg_data == 2) | (seg_data == 4)).astype(float)
        gt_et = (seg_data == 4).astype(float)

        # Load atlas mask
        atlas_file = masks_dir / f"{sample_id}_atlas_mask.nii.gz"
        if not atlas_file.exists():
            continue

        atlas_data = nib.load(str(atlas_file)).get_fdata()

        # Resize ground truth to 128x128x128 if needed
        if seg_data.shape != (128, 128, 128):
            from scipy.ndimage import zoom
            zoom_factors = [128/s for s in seg_data.shape]
            gt_tc = zoom(gt_tc, zoom_factors, order=0)
            gt_wt = zoom(gt_wt, zoom_factors, order=0)
            gt_et = zoom(gt_et, zoom_factors, order=0)

        # Calculate overlap
        for i, (channel, gt) in enumerate([('TC', gt_tc), ('WT', gt_wt), ('ET', gt_et)]):
            if gt.sum() > 0:  # Only if there's actual tumor
                # What percentage of ground truth is covered by atlas mask?
                overlap = (gt * (atlas_data[i] > 0)).sum() / gt.sum() * 100
                overlaps[channel].append(overlap)

        # Print sample result
        tc_o = overlaps['TC'][-1] if overlaps['TC'] else 0
        wt_o = overlaps['WT'][-1] if overlaps['WT'] else 0
        et_o = overlaps['ET'][-1] if overlaps['ET'] else 0
        print(f"    {sample_id}: TC={tc_o:5.1f}%, WT={wt_o:5.1f}%, ET={et_o:5.1f}%")

    # Overall statistics
    print(f"\n  Ground Truth Overlap Statistics:")
    print(f"  (What % of actual tumor is covered by atlas masks)")
    for channel in ['TC', 'WT', 'ET']:
        if overlaps[channel]:
            mean_overlap = np.mean(overlaps[channel])
            print(f"    {channel}:")
            print(f"      Mean:   {mean_overlap:6.2f}%")
            print(f"      Median: {np.median(overlaps[channel]):6.2f}%")
            print(f"      Min:    {np.min(overlaps[channel]):6.2f}%")
            print(f"      Max:    {np.max(overlaps[channel]):6.2f}%")

            # Interpretation
            if mean_overlap > 80:
                print(f"      ✓ EXCELLENT - masks cover most tumors")
            elif mean_overlap > 60:
                print(f"      ✓ GOOD - reasonable coverage")
            elif mean_overlap > 40:
                print(f"      ~ MODERATE - some tumors missed")
            else:
                print(f"      ⚠️  POOR - many tumors outside masks")

    return overlaps


def overall_assessment(coverage_stats, overlap_stats):
    """Provide overall assessment."""
    print("\n" + "="*80)
    print("5. OVERALL ASSESSMENT")
    print("="*80)

    if coverage_stats is None or overlap_stats is None:
        print("\n  Cannot provide assessment - missing data")
        return

    print("\n  Coverage (how much brain is included in masks):")
    for channel in ['TC', 'WT', 'ET']:
        mean_cov = np.mean(coverage_stats[channel])
        print(f"    {channel}: {mean_cov:.2f}%", end="")
        if mean_cov < 2:
            print(" - Very specific (may miss tumors)")
        elif mean_cov < 10:
            print(" - Good specificity ✓")
        elif mean_cov < 25:
            print(" - Moderate specificity")
        else:
            print(" - Too broad (not helpful)")

    print("\n  Overlap (how much tumor is covered):")
    for channel in ['TC', 'WT', 'ET']:
        if overlap_stats[channel]:
            mean_overlap = np.mean(overlap_stats[channel])
            print(f"    {channel}: {mean_overlap:.2f}%", end="")
            if mean_overlap > 80:
                print(" - Excellent coverage ✓✓")
            elif mean_overlap > 60:
                print(" - Good coverage ✓")
            elif mean_overlap > 40:
                print(" - Moderate (some improvement needed)")
            else:
                print(" - Poor (masks not helpful)")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    # Check if spatial prompting is likely to help
    avg_coverage = np.mean([np.mean(coverage_stats[ch]) for ch in ['TC', 'WT', 'ET']])
    avg_overlap = np.mean([np.mean(overlap_stats[ch]) for ch in ['TC', 'WT', 'ET'] if overlap_stats[ch]])

    if avg_overlap > 60 and 5 < avg_coverage < 20:
        print("\n  ✓✓ PROCEED WITH TRAINING")
        print("  The AAL atlas masks look good:")
        print(f"    - Coverage: {avg_coverage:.1f}% (specific enough)")
        print(f"    - GT Overlap: {avg_overlap:.1f}% (covers most tumors)")
        print("\n  Expected benefits:")
        print("    - Should suppress false positives outside reported regions")
        print("    - Learned alpha should be >0.5")
        print("    - Potential 2-5% Dice improvement")

    elif avg_overlap > 40:
        print("\n  ~ CAUTIOUSLY PROCEED")
        print("  The masks are okay but not optimal:")
        print(f"    - GT Overlap: {avg_overlap:.1f}%")
        print("\n  The network may learn to partially ignore the masks")
        print("  Compare results with and without spatial prompting")

    else:
        print("\n  ⚠️  DO NOT USE SPATIAL PROMPTING YET")
        print("  The masks are not good enough:")
        print(f"    - GT Overlap: {avg_overlap:.1f}% (too low)")
        print("\n  Recommendations:")
        print("    - Check volumetric_extractions.json for accuracy")
        print("    - Consider using a different atlas")
        print("    - Train baseline without spatial prompting first")


def visualize_masks_vs_gt(num_samples=15):
    """Create visualization comparing atlas masks vs ground truth."""
    print("\n" + "="*80)
    print("6. CREATING VISUALIZATIONS (ATLAS MASKS VS GROUND TRUTH)")
    print("="*80)

    data_dir = Path("/Disk1/afrouz/Data/Merged")
    masks_dir = Path("/Disk1/afrouz/Data/TextBraTS_atlas_masks_harvard-oxford_v2")
    output_pdf = Path("/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/debugging/atlas_vs_gt_visualization_harvard-oxford_v2.pdf")

    if not masks_dir.exists():
        print(f"  ⚠️  Masks directory not found")
        return

    # Get sample directories
    sample_dirs = [d for d in data_dir.iterdir() if d.is_dir() and 'BraTS' in d.name]
    sample_dirs = random.sample(sample_dirs, min(num_samples, len(sample_dirs)))

    print(f"\n  Creating visualizations for {len(sample_dirs)} samples...")
    print(f"  Output: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        for idx, sample_dir in enumerate(sample_dirs, 1):
            sample_id = sample_dir.name

            # Load ground truth
            seg_files = list(sample_dir.glob("*_seg.nii*"))
            if len(seg_files) == 0:
                continue

            seg_data = nib.load(seg_files[0]).get_fdata()

            # Convert to 3-channel format
            gt_tc = ((seg_data == 1) | (seg_data == 4)).astype(float)
            gt_wt = ((seg_data == 1) | (seg_data == 2) | (seg_data == 4)).astype(float)
            gt_et = (seg_data == 4).astype(float)

            # Load atlas mask
            atlas_file = masks_dir / f"{sample_id}_atlas_mask.nii.gz"
            if not atlas_file.exists():
                continue

            atlas_data = nib.load(str(atlas_file)).get_fdata()

            # Load FLAIR for reference
            flair_files = list(sample_dir.glob("*_flair.nii*"))
            if len(flair_files) > 0:
                flair_data = nib.load(flair_files[0]).get_fdata()
            else:
                flair_data = None

            # Resize to 128x128x128
            if seg_data.shape != (128, 128, 128):
                from scipy.ndimage import zoom
                zoom_factors = [128/s for s in seg_data.shape]
                gt_tc = zoom(gt_tc, zoom_factors, order=0)
                gt_wt = zoom(gt_wt, zoom_factors, order=0)
                gt_et = zoom(gt_et, zoom_factors, order=0)
                if flair_data is not None:
                    flair_data = zoom(flair_data, zoom_factors, order=1)

            # Find a slice with tumor
            tumor_slices = np.where(gt_wt.sum(axis=(0, 1)) > 0)[0]
            if len(tumor_slices) == 0:
                continue
            mid_slice = tumor_slices[len(tumor_slices) // 2]

            # Create figure
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.suptitle(f'{sample_id} - Slice {mid_slice}', fontsize=16, fontweight='bold')

            # Row 1: FLAIR + Ground Truth
            if flair_data is not None:
                axes[0, 0].imshow(flair_data[:, :, mid_slice].T, cmap='gray', origin='lower')
                axes[0, 0].set_title('FLAIR Image')
            else:
                axes[0, 0].text(0.5, 0.5, 'FLAIR\nNot Available', ha='center', va='center')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(gt_wt[:, :, mid_slice].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
            axes[0, 1].set_title('Ground Truth WT')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(gt_et[:, :, mid_slice].T, cmap='Blues', origin='lower', vmin=0, vmax=1)
            axes[0, 2].set_title('Ground Truth ET')
            axes[0, 2].axis('off')

            # Row 2: Atlas Masks
            axes[1, 0].imshow(atlas_data[0, :, :, mid_slice].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
            axes[1, 0].set_title('Atlas Mask TC')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(atlas_data[1, :, :, mid_slice].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
            axes[1, 1].set_title('Atlas Mask WT')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(atlas_data[2, :, :, mid_slice].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
            axes[1, 2].set_title('Atlas Mask ET')
            axes[1, 2].axis('off')

            # Row 3: Overlays
            # TC overlay
            if flair_data is not None:
                axes[2, 0].imshow(flair_data[:, :, mid_slice].T, cmap='gray', origin='lower', alpha=0.7)
            axes[2, 0].imshow(gt_tc[:, :, mid_slice].T, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
            axes[2, 0].imshow(atlas_data[0, :, :, mid_slice].T, cmap='Greens', origin='lower', alpha=0.3, vmin=0, vmax=1)
            axes[2, 0].set_title('TC: Red=GT, Green=Atlas')
            axes[2, 0].axis('off')

            # WT overlay
            if flair_data is not None:
                axes[2, 1].imshow(flair_data[:, :, mid_slice].T, cmap='gray', origin='lower', alpha=0.7)
            axes[2, 1].imshow(gt_wt[:, :, mid_slice].T, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
            axes[2, 1].imshow(atlas_data[1, :, :, mid_slice].T, cmap='Greens', origin='lower', alpha=0.3, vmin=0, vmax=1)
            axes[2, 1].set_title('WT: Red=GT, Green=Atlas')
            axes[2, 1].axis('off')

            # ET overlay
            if flair_data is not None:
                axes[2, 2].imshow(flair_data[:, :, mid_slice].T, cmap='gray', origin='lower', alpha=0.7)
            axes[2, 2].imshow(gt_et[:, :, mid_slice].T, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
            axes[2, 2].imshow(atlas_data[2, :, :, mid_slice].T, cmap='Greens', origin='lower', alpha=0.3, vmin=0, vmax=1)
            axes[2, 2].set_title('ET: Red=GT, Green=Atlas')
            axes[2, 2].axis('off')

            # Calculate overlap stats for this sample
            tc_overlap = (gt_tc * (atlas_data[0] > 0)).sum() / gt_tc.sum() * 100 if gt_tc.sum() > 0 else 0
            wt_overlap = (gt_wt * (atlas_data[1] > 0)).sum() / gt_wt.sum() * 100 if gt_wt.sum() > 0 else 0
            et_overlap = (gt_et * (atlas_data[2] > 0)).sum() / gt_et.sum() * 100 if gt_et.sum() > 0 else 0

            fig.text(0.5, 0.02, f'GT Overlap: TC={tc_overlap:.1f}%, WT={wt_overlap:.1f}%, ET={et_overlap:.1f}%',
                    ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            print(f"    [{idx}/{len(sample_dirs)}] {sample_id}: WT overlap = {wt_overlap:.1f}%")

    print(f"\n  ✓ Visualization saved to: {output_pdf}")
    print(f"  Open it to visually inspect atlas mask quality")


def main():
    print("="*80)
    print("COMPREHENSIVE HARVARD-OXFORD ATLAS DEBUGGING")
    print("="*80)
    print()

    # Run all checks
    unique_labels = check_atlas()
    region_mapping = check_region_mapping(unique_labels)
    coverage_stats = check_sample_masks(num_samples=20)
    overlap_stats = check_ground_truth_overlap(num_samples=30)
    overall_assessment(coverage_stats, overlap_stats)

    # Create visualizations
    visualize_masks_vs_gt(num_samples=15)

    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  1. debugging/debug_aal_comprehensive.txt (this output)")
    print("  2. debugging/atlas_vs_gt_visualization.pdf (visual comparison)")


if __name__ == "__main__":
    main()
