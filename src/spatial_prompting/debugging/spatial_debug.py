#!/usr/bin/env python3
"""
Simplified Spatial Prompting Debug Tool for TextBraTS

Generates a single comprehensive PDF report with:
- Visualizations of all samples (atlas masks + ground truth)
- Overlap metrics (per sample and overall)
- Learned alpha value from checkpoint

Usage:
    python spatial_debug.py --data-dir /path/to/Merged \\
                           --atlas-masks-dir /path/to/masks \\
                           --checkpoint-path /path/to/model.pt \\
                           --output report.pdf \\
                           --num-samples 20
"""

import argparse
import random
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from matplotlib.backends.backend_pdf import PdfPages

# Default paths
DEFAULT_DATA_DIR = "/Disk1/afrouz/Data/Merged"
DEFAULT_ATLAS_MASKS_DIR = "/Disk1/afrouz/Data/TextBraTS_atlas_masks_aal_v3"
DEFAULT_CHECKPOINT_PATH = "/Disk1/afrouz/Projects/TextBraTS/runs/TextBraTS_conda_spatial_prompting_aal_v31/model.pt"
DEFAULT_OUTPUT_DIR = "/Disk1/afrouz/Projects/TextBraTS/src/spatial_prompting/debugging/outputs"


def load_learned_alpha(checkpoint_path):
    """Load learned alpha value from checkpoint."""
    if not Path(checkpoint_path).exists():
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Try different possible keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Look for alpha parameter
        for key in state_dict.keys():
            if 'spatial_prompt_alpha' in key or 'alpha' in key:
                alpha_tensor = state_dict[key]
                if isinstance(alpha_tensor, torch.Tensor):
                    return alpha_tensor.item()
                return float(alpha_tensor)

        return None
    except Exception as e:
        print(f"Warning: Could not load alpha from checkpoint: {e}")
        return None


def interpret_alpha(alpha):
    """Interpret alpha value."""
    if alpha is None:
        return "Unknown"
    elif alpha > 0.8:
        return f"{alpha:.3f} - Strongly trusts masks"
    elif alpha > 0.5:
        return f"{alpha:.3f} - Moderately trusts masks"
    elif alpha > 0.2:
        return f"{alpha:.3f} - Weakly trusts masks"
    else:
        return f"{alpha:.3f} - Ignoring masks"


def calculate_overlap(gt, atlas_mask):
    """Calculate overlap percentage between ground truth and atlas mask."""
    if gt.sum() == 0:
        return 0.0
    overlap = (gt * (atlas_mask > 0)).sum() / gt.sum() * 100
    return overlap


def interpret_overlap(overlap):
    """Interpret overlap percentage."""
    if overlap > 80:
        return "✓✓ EXCELLENT"
    elif overlap > 60:
        return "✓ GOOD"
    elif overlap > 40:
        return "~ MODERATE"
    else:
        return "⚠️  POOR"


def generate_comprehensive_report(data_dir, atlas_masks_dir, checkpoint_path, output_path, num_samples, seed):
    """Generate comprehensive PDF report with visualizations and metrics."""

    data_dir = Path(data_dir)
    atlas_masks_dir = Path(atlas_masks_dir)
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SPATIAL PROMPTING COMPREHENSIVE DEBUG REPORT")
    print("="*80)

    # Load learned alpha
    print("\n1. Loading learned alpha from checkpoint...")
    alpha = load_learned_alpha(checkpoint_path)
    alpha_interpretation = interpret_alpha(alpha)
    print(f"   Learned Alpha: {alpha_interpretation}")

    # Get all samples
    print("\n2. Loading samples...")
    sample_dirs = [d for d in data_dir.iterdir() if d.is_dir() and 'BraTS' in d.name]

    if len(sample_dirs) == 0:
        print(f"   No samples found in {data_dir}")
        return

    # Sample random samples
    if seed is not None:
        random.seed(seed)

    if num_samples and num_samples < len(sample_dirs):
        sample_dirs = random.sample(sample_dirs, num_samples)

    print(f"   Found {len(sample_dirs)} samples to process")

    # Process samples and collect metrics
    print("\n3. Processing samples and generating visualizations...")
    overlap_stats = {'TC': [], 'WT': [], 'ET': []}
    sample_metrics = []

    with PdfPages(output_path) as pdf:
        # First page: Summary metrics
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Spatial Prompting Debug Report - Summary', fontsize=16, fontweight='bold')

        # We'll update this after processing all samples
        summary_ax = fig.add_subplot(111)
        summary_ax.axis('off')

        # Placeholder text
        summary_text = f"""
PROCESSING {len(sample_dirs)} SAMPLES...

Learned Alpha: {alpha_interpretation}

Overlap metrics will be calculated...
        """
        summary_ax.text(0.5, 0.5, summary_text, ha='center', va='center',
                       fontsize=12, family='monospace')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Process each sample
        for idx, sample_dir in enumerate(sample_dirs, 1):
            sample_id = sample_dir.name

            try:
                # Load ground truth
                seg_files = list(sample_dir.glob("*_seg.nii*"))
                if len(seg_files) == 0:
                    print(f"   [{idx}/{len(sample_dirs)}] Skipping {sample_id}: No segmentation found")
                    continue

                seg_data = nib.load(seg_files[0]).get_fdata()

                # Load atlas mask
                atlas_file = atlas_masks_dir / f"{sample_id}_atlas_mask.nii.gz"
                if not atlas_file.exists():
                    print(f"   [{idx}/{len(sample_dirs)}] Skipping {sample_id}: No atlas mask found")
                    continue

                atlas_data = nib.load(atlas_file).get_fdata()

                # Load FLAIR for reference
                flair_files = list(sample_dir.glob("*_flair.nii*"))
                if len(flair_files) > 0:
                    flair_data = nib.load(flair_files[0]).get_fdata()
                else:
                    flair_data = None

                # Convert GT to 3-channel format
                gt_tc = ((seg_data == 1) | (seg_data == 4)).astype(float)
                gt_wt = ((seg_data == 1) | (seg_data == 2) | (seg_data == 4)).astype(float)
                gt_et = (seg_data == 4).astype(float)

                # Resize GT to 128x128x128 if needed
                if seg_data.shape != (128, 128, 128):
                    zoom_factors = [128/s for s in seg_data.shape]
                    gt_tc = zoom(gt_tc, zoom_factors, order=0)
                    gt_wt = zoom(gt_wt, zoom_factors, order=0)
                    gt_et = zoom(gt_et, zoom_factors, order=0)
                    if flair_data is not None:
                        flair_data = zoom(flair_data, zoom_factors, order=1)

                # Calculate overlap metrics for this sample
                overlap_tc = calculate_overlap(gt_tc, atlas_data[0])
                overlap_wt = calculate_overlap(gt_wt, atlas_data[1])
                overlap_et = calculate_overlap(gt_et, atlas_data[2])

                overlap_stats['TC'].append(overlap_tc)
                overlap_stats['WT'].append(overlap_wt)
                overlap_stats['ET'].append(overlap_et)

                sample_metrics.append({
                    'sample_id': sample_id,
                    'TC': overlap_tc,
                    'WT': overlap_wt,
                    'ET': overlap_et
                })

                # Print to terminal
                print(f"   [{idx}/{len(sample_dirs)}] {sample_id}: "
                      f"TC={overlap_tc:5.1f}%, WT={overlap_wt:5.1f}%, ET={overlap_et:5.1f}%")

                # Find a slice with tumor
                tumor_slices = np.where(gt_wt.sum(axis=(0, 1)) > 0)[0]
                if len(tumor_slices) == 0:
                    mid_slice = 64
                else:
                    mid_slice = tumor_slices[len(tumor_slices) // 2]

                # Create visualization page
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))

                # Title with overlap metrics
                title = f'{sample_id} - Slice {mid_slice}\n'
                title += f'Overlap: TC={overlap_tc:.1f}% {interpret_overlap(overlap_tc)}, '
                title += f'WT={overlap_wt:.1f}% {interpret_overlap(overlap_wt)}, '
                title += f'ET={overlap_et:.1f}% {interpret_overlap(overlap_et)}'
                fig.suptitle(title, fontsize=14, fontweight='bold')

                # Row 1: FLAIR + Ground Truth (WT, TC)
                if flair_data is not None:
                    axes[0, 0].imshow(flair_data[:, :, mid_slice].T, cmap='gray', origin='lower')
                    axes[0, 0].set_title('FLAIR')
                else:
                    axes[0, 0].text(0.5, 0.5, 'FLAIR\nNot Available', ha='center', va='center')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(gt_wt[:, :, mid_slice].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
                axes[0, 1].set_title('GT: Whole Tumor (WT)')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(gt_tc[:, :, mid_slice].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
                axes[0, 2].set_title('GT: Tumor Core (TC)')
                axes[0, 2].axis('off')

                # Row 2: Atlas Masks (WT, TC, ET)
                axes[1, 0].imshow(atlas_data[1, :, :, mid_slice].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
                axes[1, 0].set_title('Atlas Mask: WT')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(atlas_data[0, :, :, mid_slice].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
                axes[1, 1].set_title('Atlas Mask: TC')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(atlas_data[2, :, :, mid_slice].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
                axes[1, 2].set_title('Atlas Mask: ET')
                axes[1, 2].axis('off')

                # Row 3: Overlays (Red=GT, Green=Atlas)
                axes[2, 0].imshow(gt_wt[:, :, mid_slice].T, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
                axes[2, 0].imshow(atlas_data[1, :, :, mid_slice].T, cmap='Greens', origin='lower', alpha=0.5, vmin=0, vmax=1)
                axes[2, 0].set_title('Overlay: WT (Red=GT, Green=Atlas)')
                axes[2, 0].axis('off')

                axes[2, 1].imshow(gt_tc[:, :, mid_slice].T, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
                axes[2, 1].imshow(atlas_data[0, :, :, mid_slice].T, cmap='Greens', origin='lower', alpha=0.5, vmin=0, vmax=1)
                axes[2, 1].set_title('Overlay: TC (Red=GT, Green=Atlas)')
                axes[2, 1].axis('off')

                axes[2, 2].imshow(gt_et[:, :, mid_slice].T, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
                axes[2, 2].imshow(atlas_data[2, :, :, mid_slice].T, cmap='Greens', origin='lower', alpha=0.5, vmin=0, vmax=1)
                axes[2, 2].set_title('Overlay: ET (Red=GT, Green=Atlas)')
                axes[2, 2].axis('off')

                plt.tight_layout()
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt.close(fig)

            except Exception as e:
                print(f"   [{idx}/{len(sample_dirs)}] Error processing {sample_id}: {e}")
                continue

        # Final page: Overall summary statistics
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Spatial Prompting Debug Report - Overall Summary', fontsize=16, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.axis('off')

        # Calculate statistics
        summary_lines = []
        summary_lines.append("="*70)
        summary_lines.append("OVERALL OVERLAP STATISTICS")
        summary_lines.append("="*70)
        summary_lines.append("")
        summary_lines.append(f"Samples Processed: {len(sample_metrics)}")
        summary_lines.append("")

        for channel in ['TC', 'WT', 'ET']:
            if overlap_stats[channel]:
                mean_val = np.mean(overlap_stats[channel])
                median_val = np.median(overlap_stats[channel])
                min_val = np.min(overlap_stats[channel])
                max_val = np.max(overlap_stats[channel])

                summary_lines.append(f"{channel}:")
                summary_lines.append(f"  Mean:   {mean_val:6.2f}%  {interpret_overlap(mean_val)}")
                summary_lines.append(f"  Median: {median_val:6.2f}%")
                summary_lines.append(f"  Min:    {min_val:6.2f}%")
                summary_lines.append(f"  Max:    {max_val:6.2f}%")
                summary_lines.append("")

        summary_lines.append("="*70)
        summary_lines.append("LEARNED ALPHA")
        summary_lines.append("="*70)
        summary_lines.append("")
        summary_lines.append(f"Alpha: {alpha_interpretation}")
        summary_lines.append("")

        # Overall recommendation
        avg_overlap = np.mean([np.mean(overlap_stats[ch]) for ch in ['TC', 'WT', 'ET'] if overlap_stats[ch]])
        summary_lines.append("="*70)
        summary_lines.append("RECOMMENDATION")
        summary_lines.append("="*70)
        summary_lines.append("")

        if avg_overlap > 60:
            summary_lines.append("✓✓ PROCEED WITH TRAINING")
            summary_lines.append("   The atlas masks align well with ground truth")
        elif avg_overlap > 40:
            summary_lines.append("~ CAUTIOUSLY PROCEED")
            summary_lines.append("   The masks are okay but not optimal")
        else:
            summary_lines.append("⚠️  DO NOT USE SPATIAL PROMPTING YET")
            summary_lines.append("   The masks do not align well with ground truth")

        summary_text = '\n'.join(summary_lines)
        ax.text(0.1, 0.95, summary_text, ha='left', va='top',
               fontsize=11, family='monospace', transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # Print final summary to terminal
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"\nSamples Processed: {len(sample_metrics)}")
    print(f"\nOverlap Statistics:")

    for channel in ['TC', 'WT', 'ET']:
        if overlap_stats[channel]:
            mean_val = np.mean(overlap_stats[channel])
            print(f"  {channel}: Mean={mean_val:6.2f}% {interpret_overlap(mean_val)}")

    print(f"\nLearned Alpha: {alpha_interpretation}")
    print(f"\n✓ Report saved to: {output_path}")
    print(f"  Total pages: {len(sample_metrics) + 2}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive spatial prompting debug report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f'Directory containing BraTS data (default: {DEFAULT_DATA_DIR})'
    )
    parser.add_argument(
        '--atlas-masks-dir',
        type=str,
        default=DEFAULT_ATLAS_MASKS_DIR,
        help=f'Directory containing atlas masks (default: {DEFAULT_ATLAS_MASKS_DIR})'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f'Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=f'{DEFAULT_OUTPUT_DIR}/spatial_debug_report.pdf',
        help=f'Output PDF path (default: {DEFAULT_OUTPUT_DIR}/spatial_debug_report.pdf)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    args = parser.parse_args()

    generate_comprehensive_report(
        data_dir=args.data_dir,
        atlas_masks_dir=args.atlas_masks_dir,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
