"""
Visualize BraTS samples with all modalities and segmentation masks in a PDF.

This script creates a multi-page PDF where each page shows:
- All 4 MRI modalities (FLAIR, T1, T1CE, T2)
- Segmentation mask overlay on each modality
- Middle slice visualization for quick overview

Usage:
    python visualize_samples_pdf.py --data_dir /path/to/data --output_pdf output.pdf --num_samples 10
"""

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from tqdm import tqdm
import matplotlib.patches as mpatches


def load_sample(case_dir, case_name):
    """
    Load all modalities and segmentation for a single case.

    Args:
        case_dir: Path to case directory
        case_name: Case name (e.g., 'BraTS20_Training_001')

    Returns:
        Dictionary with modality data and segmentation
    """
    data = {}

    # Load all four modalities
    modalities = ['flair', 't1', 't1ce', 't2']
    for modality in modalities:
        mod_path = case_dir / f"{case_name}_{modality}.nii"
        if mod_path.exists():
            nii = nib.load(str(mod_path))
            data[modality] = nii.get_fdata()
        else:
            print(f"Warning: Missing {modality} for {case_name}")
            data[modality] = None

    # Load segmentation
    seg_path = case_dir / f"{case_name}_seg.nii"
    if seg_path.exists():
        seg_nii = nib.load(str(seg_path))
        data['seg'] = seg_nii.get_fdata().astype(np.uint8)
    else:
        print(f"Warning: Missing segmentation for {case_name}")
        data['seg'] = None

    return data


def create_overlay(image_slice, seg_slice, alpha=0.4):
    """
    Create segmentation overlay on image slice.

    Args:
        image_slice: 2D image array
        seg_slice: 2D segmentation mask
        alpha: Transparency of overlay

    Returns:
        RGB overlay image
    """
    # Normalize image to 0-1
    img_min, img_max = image_slice.min(), image_slice.max()
    if img_max > img_min:
        img_norm = (image_slice - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(image_slice)

    # Create RGB image
    overlay = np.stack([img_norm] * 3, axis=-1)

    # Define colors for each label
    colors = {
        1: np.array([1.0, 0.0, 0.0]),  # NCR/NET - Red
        2: np.array([0.0, 1.0, 0.0]),  # Edema - Green
        4: np.array([0.0, 0.0, 1.0]),  # Enhancing - Blue
    }

    # Apply colored overlay
    for label, color in colors.items():
        mask = (seg_slice == label)
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    return overlay


def visualize_case(data, case_name, slice_idx=None):
    """
    Create visualization for a single case.

    Args:
        data: Dictionary with modality data
        case_name: Case name for title
        slice_idx: Slice index to visualize (default: middle slice)

    Returns:
        matplotlib figure
    """
    modalities = ['flair', 't1', 't1ce', 't2']

    # Get middle slice if not specified
    if slice_idx is None and data['flair'] is not None:
        slice_idx = data['flair'].shape[2] // 2

    # Create figure with 2 rows x 4 columns
    # Row 1: Original modalities
    # Row 2: Modalities with segmentation overlay
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Add sample name as main title
    fig.suptitle(f"Sample: {case_name}", fontsize=18, fontweight='bold', y=0.98)

    # Add slice info as subtitle
    fig.text(0.5, 0.94, f"Slice Index: {slice_idx}", ha='center', fontsize=12, style='italic')

    for idx, modality in enumerate(modalities):
        # Get data
        if data[modality] is not None:
            img_slice = data[modality][:, :, slice_idx]
        else:
            img_slice = np.zeros((128, 128))

        if data['seg'] is not None:
            seg_slice = data['seg'][:, :, slice_idx]
        else:
            seg_slice = np.zeros_like(img_slice)

        # Top row: Original modality
        axes[0, idx].imshow(img_slice, cmap='gray', aspect='auto')
        axes[0, idx].set_title(modality.upper(), fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')

        # Bottom row: Overlay
        overlay = create_overlay(img_slice, seg_slice, alpha=0.5)
        axes[1, idx].imshow(overlay, aspect='auto')
        axes[1, idx].set_title(f"{modality.upper()} + Seg", fontsize=12)
        axes[1, idx].axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='red', label='NCR/NET (Label 1)'),
        mpatches.Patch(facecolor='green', label='Edema (Label 2)'),
        mpatches.Patch(facecolor='blue', label='Enhancing (Label 4)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize BraTS samples in PDF")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Disk1/afrouz/Data/Merged',
        help='Path to BraTS data directory'
    )
    parser.add_argument(
        '--output_pdf',
        type=str,
        default='brats_samples_visualization.pdf',
        help='Output PDF filename'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of samples to visualize (default: 10, use -1 for all)'
    )
    parser.add_argument(
        '--slice_idx',
        type=int,
        default=None,
        help='Specific slice index to visualize (default: middle slice)'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_path = Path(args.output_pdf)

    # Get case directories
    case_dirs = sorted([d for d in data_dir.iterdir()
                       if d.is_dir() and d.name.startswith('BraTS')])

    # Limit number of samples
    if args.num_samples > 0:
        case_dirs = case_dirs[:args.num_samples]

    print(f"Visualizing {len(case_dirs)} samples...")
    print(f"Output PDF: {output_path}")

    # Create PDF
    with PdfPages(output_path) as pdf:
        for case_dir in tqdm(case_dirs, desc="Processing cases"):
            case_name = case_dir.name

            try:
                # Load data
                data = load_sample(case_dir, case_name)

                # Skip if essential data is missing
                if data['flair'] is None or data['seg'] is None:
                    print(f"Skipping {case_name}: missing essential data")
                    continue

                # Create visualization
                fig = visualize_case(data, case_name, slice_idx=args.slice_idx)

                # Save to PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            except Exception as e:
                print(f"Error processing {case_name}: {e}")
                continue

    print(f"\nVisualization complete!")
    print(f"PDF saved to: {output_path}")
    print(f"Total pages: {len(case_dirs)}")


if __name__ == '__main__':
    main()
