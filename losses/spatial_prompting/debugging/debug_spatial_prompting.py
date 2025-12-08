"""
Debug script to analyze spatial prompting effectiveness
"""

import torch
import nibabel as nib
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def check_atlas_coverage(atlas_masks_dir="/Disk1/afrouz/Data/TextBraTS_atlas_masks"):
    """Check how restrictive atlas masks are"""
    atlas_dir = Path(atlas_masks_dir)
    mask_files = list(atlas_dir.glob("*_atlas_mask.nii.gz"))

    if len(mask_files) == 0:
        print(f"No atlas masks found in {atlas_masks_dir}")
        return

    stats = {'TC': [], 'WT': [], 'ET': []}
    whole_brain_count = {'TC': 0, 'WT': 0, 'ET': 0}

    print(f"Analyzing {len(mask_files)} atlas masks...")

    for mask_file in mask_files:
        mask_data = nib.load(mask_file).get_fdata()

        # Calculate coverage (percentage of brain that's allowed)
        for i, channel in enumerate(['TC', 'WT', 'ET']):
            coverage = (mask_data[i] > 0).sum() / mask_data[i].size * 100
            stats[channel].append(coverage)

            # Check if it's essentially whole brain (>95% coverage)
            if coverage > 95:
                whole_brain_count[channel] += 1

    print("\n" + "="*60)
    print("ATLAS MASK COVERAGE STATISTICS")
    print("="*60)

    for channel in ['TC', 'WT', 'ET']:
        print(f"\n{channel} Channel:")
        print(f"  Mean coverage: {np.mean(stats[channel]):.2f}%")
        print(f"  Median coverage: {np.median(stats[channel]):.2f}%")
        print(f"  Min coverage: {np.min(stats[channel]):.2f}%")
        print(f"  Max coverage: {np.max(stats[channel]):.2f}%")
        print(f"  Whole brain fallback count: {whole_brain_count[channel]}/{len(mask_files)} ({whole_brain_count[channel]/len(mask_files)*100:.1f}%)")

    # If most masks are whole brain, that's a problem
    if any(whole_brain_count[ch] > len(mask_files) * 0.7 for ch in ['TC', 'WT', 'ET']):
        print("\n" + "⚠️  WARNING: Many masks are using whole brain fallback!")
        print("   This means the spatial prompting isn't adding much constraint.")
        print("   Check your volumetric_extractions.json to see if regions are being extracted.")

    return stats


def check_learned_alpha(checkpoint_path):
    """Check what alpha value the network learned"""
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Try to find alpha in state_dict
    state_dict = checkpoint.get('state_dict', checkpoint)

    alpha_key = None
    for key in state_dict.keys():
        if 'spatial_prompt_alpha' in key:
            alpha_key = key
            break

    if alpha_key:
        alpha_value = state_dict[alpha_key].item()
        print("\n" + "="*60)
        print("LEARNED ALPHA VALUE")
        print("="*60)
        print(f"Alpha: {alpha_value:.4f}")
        print(f"\nInterpretation:")
        if alpha_value > 0.8:
            print("  → Network STRONGLY trusts atlas masks")
        elif alpha_value > 0.5:
            print("  → Network MODERATELY trusts atlas masks")
        elif alpha_value > 0.2:
            print("  → Network WEAKLY trusts atlas masks")
        else:
            print("  → Network is IGNORING atlas masks (learned they're not helpful)")

        return alpha_value
    else:
        print("Could not find spatial_prompt_alpha in checkpoint")
        return None


def analyze_mask_vs_ground_truth(data_dir, atlas_masks_dir, num_samples=10):
    """
    Check if atlas masks align well with ground truth labels
    """
    from pathlib import Path
    import random

    data_path = Path(data_dir)
    atlas_path = Path(atlas_masks_dir)

    # Get sample directories
    sample_dirs = [d for d in data_path.iterdir() if d.is_dir() and 'BraTS' in d.name]

    if len(sample_dirs) == 0:
        print(f"No BraTS samples found in {data_dir}")
        return

    # Randomly sample
    sample_dirs = random.sample(sample_dirs, min(num_samples, len(sample_dirs)))

    overlaps = {'TC': [], 'WT': [], 'ET': []}

    print("\n" + "="*60)
    print("ATLAS MASK vs GROUND TRUTH ALIGNMENT")
    print("="*60)

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name

        # Load ground truth segmentation
        seg_files = list(sample_dir.glob("*_seg.nii*"))
        if len(seg_files) == 0:
            continue

        seg_data = nib.load(seg_files[0]).get_fdata()

        # Convert to 3-channel format (TC, WT, ET) like MONAI does
        gt_tc = ((seg_data == 1) | (seg_data == 4)).astype(float)  # Tumor core
        gt_wt = ((seg_data == 1) | (seg_data == 2) | (seg_data == 4)).astype(float)  # Whole tumor
        gt_et = (seg_data == 4).astype(float)  # Enhancing tumor

        # Load atlas mask
        atlas_file = atlas_path / f"{sample_id}_atlas_mask.nii.gz"
        if not atlas_file.exists():
            continue

        atlas_data = nib.load(atlas_file).get_fdata()

        # Resize ground truth to 128x128x128 if needed (simple nearest neighbor)
        if seg_data.shape != (128, 128, 128):
            from scipy.ndimage import zoom
            zoom_factors = [128/s for s in seg_data.shape]
            gt_tc = zoom(gt_tc, zoom_factors, order=0)
            gt_wt = zoom(gt_wt, zoom_factors, order=0)
            gt_et = zoom(gt_et, zoom_factors, order=0)

        # Calculate overlap: what percentage of ground truth tumor is inside atlas mask?
        for i, (channel, gt) in enumerate([('TC', gt_tc), ('WT', gt_wt), ('ET', gt_et)]):
            if gt.sum() > 0:  # Only if there's actual tumor
                overlap = (gt * (atlas_data[i] > 0)).sum() / gt.sum() * 100
                overlaps[channel].append(overlap)

    print(f"\nAnalyzed {len(sample_dirs)} samples")
    print("\nHow much of the GROUND TRUTH tumor is covered by atlas masks:")
    for channel in ['TC', 'WT', 'ET']:
        if len(overlaps[channel]) > 0:
            print(f"\n{channel}:")
            print(f"  Mean overlap: {np.mean(overlaps[channel]):.2f}%")
            print(f"  Median overlap: {np.median(overlaps[channel]):.2f}%")
            print(f"  Min overlap: {np.min(overlaps[channel]):.2f}%")

            if np.mean(overlaps[channel]) < 80:
                print(f"  ⚠️  WARNING: Atlas masks miss significant tumor regions!")


def visualize_sample_with_mask(sample_id, data_dir, atlas_masks_dir, output_file="debug_viz.png"):
    """Visualize a sample with its atlas mask and ground truth"""
    from pathlib import Path

    sample_dir = Path(data_dir) / sample_id
    if not sample_dir.exists():
        print(f"Sample not found: {sample_dir}")
        return

    # Load ground truth
    seg_files = list(sample_dir.glob("*_seg.nii*"))
    if len(seg_files) == 0:
        print(f"No segmentation found for {sample_id}")
        return

    seg_data = nib.load(seg_files[0]).get_fdata()

    # Load atlas mask
    atlas_file = Path(atlas_masks_dir) / f"{sample_id}_atlas_mask.nii.gz"
    if not atlas_file.exists():
        print(f"No atlas mask found for {sample_id}")
        return

    atlas_data = nib.load(atlas_file).get_fdata()

    # Load FLAIR for reference
    flair_files = list(sample_dir.glob("*_flair.nii*"))
    if len(flair_files) > 0:
        flair_data = nib.load(flair_files[0]).get_fdata()
    else:
        flair_data = None

    # Plot middle slice
    mid_slice = seg_data.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Sample: {sample_id}", fontsize=16)

    # Top row: Atlas masks (resized to match original if needed)
    for i, title in enumerate(['Atlas TC', 'Atlas WT', 'Atlas ET']):
        axes[0, i].imshow(atlas_data[i, :, :, 64].T, cmap='hot', origin='lower')
        axes[0, i].set_title(title)
        axes[0, i].axis('off')

    # Bottom row: Ground truth
    gt_tc = ((seg_data[:, :, mid_slice] == 1) | (seg_data[:, :, mid_slice] == 4)).astype(float)
    gt_wt = ((seg_data[:, :, mid_slice] == 1) | (seg_data[:, :, mid_slice] == 2) | (seg_data[:, :, mid_slice] == 4)).astype(float)
    gt_et = (seg_data[:, :, mid_slice] == 4).astype(float)

    for i, (title, gt) in enumerate([('GT TC', gt_tc), ('GT WT', gt_wt), ('GT ET', gt_et)]):
        axes[1, i].imshow(gt.T, cmap='hot', origin='lower')
        axes[1, i].set_title(title)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")


if __name__ == "__main__":
    print("="*60)
    print("SPATIAL PROMPTING DIAGNOSTIC TOOL")
    print("="*60)

    # 1. Check atlas mask coverage
    print("\n1. Checking atlas mask coverage...")
    check_atlas_coverage()

    # 2. Check learned alpha (if you have a checkpoint)
    print("\n2. Checking learned alpha value...")
    checkpoint_path = "/Disk1/afrouz/Projects/TextBraTS/runs/TextBraTS_conda_spatial_prompting/model.pt"
    # Update this path to your actual checkpoint
    check_learned_alpha(checkpoint_path)

    # 3. Analyze mask-GT alignment
    print("\n3. Analyzing atlas mask vs ground truth alignment...")
    analyze_mask_vs_ground_truth(
        data_dir="/Disk1/afrouz/Data/Merged",
        atlas_masks_dir="/Disk1/afrouz/Data/TextBraTS_atlas_masks_fixed",
        num_samples=20
    )

    # 4. Visualize a sample
    print("\n4. Creating visualization...")
    visualize_sample_with_mask(
        sample_id="BraTS20_Training_001",
        data_dir="/Disk1/afrouz/Data/Merged",
        atlas_masks_dir="/Disk1/afrouz/Data/TextBraTS_atlas_masks_fixed",
        output_file="/Disk1/afrouz/Projects/TextBraTS/debug_spatial_prompting_fixed_viz.png"
    )

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
