"""
Brain Atlas Preprocessing Script for TextBraTS

This script downloads and preprocesses standard brain atlases (Harvard-Oxford or AAL)
and resamples them to the target BraTS shape (128x128x128).

The atlas serves as a spatial reference map where each voxel has an integer label
corresponding to an anatomical region (e.g., 1=Frontal Lobe, 2=Temporal Lobe, etc.)

Usage:
    python generate_brain_atlas_masks.py --data_dir ./data/TextBraTSData \
                                          --output_dir ./atlas_preprocessed \
                                          --target_shape 128 128 128 \
                                          --atlas_type harvard-oxford

Output:
    - Resampled atlas NIfTI file (128x128x128) aligned to BraTS space
    - Label mapping JSON (label_id -> region_name)
    - Visualization of atlas regions
"""

# SSL Certificate Workaround for Institutional Networks
# This disables SSL verification for atlas downloads from nilearn
# Appropriate for research environments with custom certificates
import ssl
import os

# Method 1: Disable SSL verification globally for urllib
ssl._create_default_https_context = ssl._create_unverified_context

# Method 2: Set environment variables for requests library (used by nilearn)
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Method 3: Monkey-patch urllib3 and requests before nilearn imports
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch urllib3 PoolManager to disable verification
import functools
_orig_poolmanager_init = urllib3.PoolManager.__init__

@functools.wraps(_orig_poolmanager_init)
def _patched_poolmanager_init(self, *args, **kwargs):
    kwargs['cert_reqs'] = ssl.CERT_NONE
    kwargs['assert_hostname'] = False
    return _orig_poolmanager_init(self, *args, **kwargs)

urllib3.PoolManager.__init__ = _patched_poolmanager_init

# Patch requests
import requests

_orig_session_init = requests.Session.__init__

@functools.wraps(_orig_session_init)
def _patched_session_init(self, *args, **kwargs):
    _orig_session_init(self, *args, **kwargs)
    self.verify = False

requests.Session.__init__ = _patched_session_init

# Now import other modules
import json
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse

from nilearn import datasets, image, plotting
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class BrainAtlasPreprocessor:
    """
    Downloads and preprocesses standard brain atlases for use with BraTS data.
    """

    def __init__(self, atlas_type='harvard-oxford'):
        """
        Initialize the atlas preprocessor.

        Args:
            atlas_type: Type of atlas ('harvard-oxford' or 'aal')
        """
        self.atlas_type = atlas_type
        self.atlas_img = None
        self.label_names = {}

        print(f"Initializing {atlas_type} atlas preprocessor...")
        self._load_atlas()

    def _load_atlas(self):
        """Download and load the brain atlas from nilearn."""

        if self.atlas_type == 'harvard-oxford':
            print("Downloading Harvard-Oxford atlas...")

            # Fetch cortical atlas (probabilistic or maximum probability)
            atlas_cortical = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

            # Fetch subcortical atlas
            atlas_subcortical = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')

            # The maps attribute can be either a string path or already loaded image
            # Load the NIfTI images
            if isinstance(atlas_cortical.maps, str):
                cort_img = nib.load(atlas_cortical.maps)
            else:
                cort_img = atlas_cortical.maps

            if isinstance(atlas_subcortical.maps, str):
                subcort_img = nib.load(atlas_subcortical.maps)
            else:
                subcort_img = atlas_subcortical.maps

            # Get the data arrays
            cort_data = cort_img.get_fdata()
            subcort_data = subcort_img.get_fdata()

            # Combine cortical and subcortical atlases
            # Offset subcortical labels so they don't overlap with cortical
            max_cort_label = int(np.max(cort_data))
            print(f"Cortical atlas: {max_cort_label} regions")
            print(f"Subcortical atlas: {len(atlas_subcortical.labels)} regions")

            # Offset subcortical labels
            subcort_data_offset = subcort_data.copy()
            subcort_data_offset[subcort_data > 0] += max_cort_label

            # Combine by taking maximum (assumes no overlap)
            combined_data = np.maximum(cort_data, subcort_data_offset)

            # Create combined atlas image
            self.atlas_img = nib.Nifti1Image(
                combined_data.astype(np.int16),
                cort_img.affine,
                cort_img.header
            )

            # Create combined label mapping
            self.label_names = {}

            # Add cortical labels
            for i, label in enumerate(atlas_cortical.labels):
                self.label_names[i] = f"Cortical_{label}"

            # Add subcortical labels (offset)
            for i, label in enumerate(atlas_subcortical.labels):
                if i > 0:  # Skip background
                    self.label_names[max_cort_label + i] = f"Subcortical_{label}"

            print(f"Combined atlas: {len(self.label_names)} total regions")
            print(f"Original atlas shape: {self.atlas_img.shape}")
            print(f"Original atlas resolution: {self.atlas_img.header.get_zooms()[:3]} mm")

        elif self.atlas_type == 'aal':
            print("Downloading AAL atlas...")
            atlas = datasets.fetch_atlas_aal()

            # Handle both string path and loaded image
            if isinstance(atlas.maps, str):
                self.atlas_img = nib.load(atlas.maps)
            else:
                self.atlas_img = atlas.maps

            # Create label mapping from AAL labels
            self.label_names = {}
            for i, label in enumerate(atlas.labels):
                # AAL indices start at 1
                self.label_names[i + 1] = label

            print(f"AAL atlas: {len(self.label_names)} regions")
            print(f"Original atlas shape: {self.atlas_img.shape}")

        elif self.atlas_type == 'talairach':
            print("Downloading Talairach atlas (lobe level)...")
            atlas = datasets.fetch_atlas_talairach(level_name='lobe')

            # Handle both string path and loaded image
            if isinstance(atlas.maps, str):
                self.atlas_img = nib.load(atlas.maps)
            else:
                self.atlas_img = atlas.maps

            # Create label mapping from Talairach labels
            self.label_names = {}
            for i, label in enumerate(atlas.labels):
                # Talairach atlas labels match indices
                self.label_names[i] = label

            print(f"Talairach atlas: {len(self.label_names)} regions")
            print(f"Labels: {atlas.labels}")
            print(f"Original atlas shape: {self.atlas_img.shape}")
            print(f"Original atlas resolution: {self.atlas_img.header.get_zooms()[:3]} mm")

        else:
            raise ValueError(f"Atlas type '{self.atlas_type}' not supported. Choose 'harvard-oxford', 'aal', or 'talairach'")

    def resample_to_target(self, target_img, interpolation='nearest'):
        """
        Resample the atlas to match a target image's shape and affine.

        Args:
            target_img: Target nibabel image to match
            interpolation: Interpolation method ('nearest' for label images)

        Returns:
            Resampled atlas image
        """
        print(f"Resampling atlas to target shape {target_img.shape}...")

        resampled_img = image.resample_to_img(
            self.atlas_img,
            target_img,
            interpolation=interpolation
        )

        print(f"Resampled atlas shape: {resampled_img.shape}")

        return resampled_img

    def pad_to_shape(self, target_shape=(128, 128, 128)):
        """
        Pad the atlas to a specific target shape WITHOUT interpolation.
        This preserves exact label IDs and sharp anatomical boundaries.

        Args:
            target_shape: Desired output shape (e.g., (128, 128, 128))

        Returns:
            Padded atlas image
        """
        print(f"Padding atlas to shape {target_shape}...")

        # Get original data and shape
        original_data = self.atlas_img.get_fdata()
        original_shape = np.array(original_data.shape[:3])
        target_shape = np.array(target_shape)

        print(f"Original atlas shape: {original_shape}")
        print(f"Target shape: {target_shape}")

        # Check if target shape is larger than original
        if np.any(target_shape < original_shape):
            raise ValueError(f"Target shape {target_shape} must be >= original shape {original_shape}")

        # Calculate padding needed (centered)
        pad_total = target_shape - original_shape
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before

        print(f"Padding: before={pad_before}, after={pad_after}")

        # Pad with zeros (background label)
        padded_data = np.pad(
            original_data,
            list(zip(pad_before, pad_after)),
            mode='constant',
            constant_values=0
        )

        print(f"Padded atlas shape: {padded_data.shape}")

        # Update affine matrix to account for padding
        # The origin shifts by pad_before voxels
        affine = self.atlas_img.affine.copy()
        voxel_sizes = np.array(self.atlas_img.header.get_zooms()[:3])

        # Adjust origin (translation part of affine)
        affine[:3, 3] -= pad_before * voxel_sizes * np.sign(np.diag(affine[:3, :3]))

        # Create new NIfTI image
        padded_img = nib.Nifti1Image(padded_data.astype(np.int16), affine)

        # Verify label preservation
        original_labels = set(np.unique(original_data))
        padded_labels = set(np.unique(padded_data))
        assert original_labels == padded_labels, "Labels changed during padding!"

        print(f"✓ Label preservation verified: {len(original_labels)} unique labels")

        return padded_img

    def resample_to_shape(self, target_shape=(128, 128, 128), target_affine=None):
        """
        Resample the atlas to a specific target shape.

        Args:
            target_shape: Desired output shape (e.g., (128, 128, 128))
            target_affine: Target affine matrix. If None, uses scaled version of original

        Returns:
            Resampled atlas image
        """
        print(f"Resampling atlas to shape {target_shape}...")

        # If no target affine provided, create one with scaled voxel sizes
        if target_affine is None:
            original_shape = np.array(self.atlas_img.shape[:3])
            new_shape = np.array(target_shape)

            # Calculate new voxel sizes
            original_voxel_sizes = np.array(self.atlas_img.header.get_zooms()[:3])
            scaling_factors = original_shape / new_shape
            new_voxel_sizes = original_voxel_sizes * scaling_factors

            # Create new affine with scaled voxel sizes
            target_affine = self.atlas_img.affine.copy()
            target_affine[0, 0] = new_voxel_sizes[0] if target_affine[0, 0] > 0 else -new_voxel_sizes[0]
            target_affine[1, 1] = new_voxel_sizes[1] if target_affine[1, 1] > 0 else -new_voxel_sizes[1]
            target_affine[2, 2] = new_voxel_sizes[2] if target_affine[2, 2] > 0 else -new_voxel_sizes[2]

            print(f"Original voxel size: {original_voxel_sizes} mm")
            print(f"New voxel size: {new_voxel_sizes} mm")

        # Create target image with desired shape
        target_data = np.zeros(target_shape)
        target_img = nib.Nifti1Image(target_data, target_affine)

        # Resample
        resampled_img = self.resample_to_target(target_img, interpolation='nearest')

        return resampled_img

    def save_atlas(self, output_path, resampled_atlas_img, method='resampled'):
        """
        Save the resampled atlas and its label mapping.

        Args:
            output_path: Output directory path
            resampled_atlas_img: Resampled atlas NIfTI image
            method: Processing method used ('resampled' or 'padded')
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save atlas NIfTI
        atlas_file = output_path / f'brain_atlas_{self.atlas_type}_{method}.nii.gz'
        nib.save(resampled_atlas_img, str(atlas_file))
        print(f"Saved {method} atlas to: {atlas_file}")

        # Save label mapping as JSON
        labels_file = output_path / f'atlas_labels_{self.atlas_type}.json'
        with open(labels_file, 'w') as f:
            # Convert integer keys to strings for JSON
            json_labels = {str(k): v for k, v in self.label_names.items()}
            json.dump(json_labels, f, indent=2)
        print(f"Saved label mapping to: {labels_file}")

        # Save atlas statistics
        data = resampled_atlas_img.get_fdata()
        unique_labels = np.unique(data)

        stats = {
            'atlas_type': self.atlas_type,
            'shape': list(resampled_atlas_img.shape),
            'voxel_size_mm': [float(x) for x in resampled_atlas_img.header.get_zooms()[:3]],
            'num_regions': int(len(unique_labels) - 1),  # Exclude background
            'unique_labels': [int(x) for x in unique_labels],
            'total_voxels': int(np.prod(data.shape)),
            'background_voxels': int(np.sum(data == 0)),
        }

        stats_file = output_path / f'atlas_stats_{self.atlas_type}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved atlas statistics to: {stats_file}")

        return atlas_file, labels_file, stats_file

    def visualize_atlas(self, resampled_atlas_img, output_path, num_slices=5):
        """
        Create visualization of the atlas.

        Args:
            resampled_atlas_img: Resampled atlas image
            output_path: Directory to save visualizations
            num_slices: Number of slices to show per axis
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Creating atlas visualizations...")

        # Create multi-slice visualization
        fig, axes = plt.subplots(3, num_slices, figsize=(15, 9))
        fig.suptitle(f'{self.atlas_type.upper()} Brain Atlas - Resampled', fontsize=16)

        data = resampled_atlas_img.get_fdata()

        # Axial slices
        for i, slice_idx in enumerate(np.linspace(20, data.shape[2]-20, num_slices, dtype=int)):
            axes[0, i].imshow(data[:, :, slice_idx].T, cmap='tab20', origin='lower')
            axes[0, i].set_title(f'Axial {slice_idx}')
            axes[0, i].axis('off')

        # Coronal slices
        for i, slice_idx in enumerate(np.linspace(20, data.shape[1]-20, num_slices, dtype=int)):
            axes[1, i].imshow(data[:, slice_idx, :].T, cmap='tab20', origin='lower')
            axes[1, i].set_title(f'Coronal {slice_idx}')
            axes[1, i].axis('off')

        # Sagittal slices
        for i, slice_idx in enumerate(np.linspace(20, data.shape[0]-20, num_slices, dtype=int)):
            axes[2, i].imshow(data[slice_idx, :, :].T, cmap='tab20', origin='lower')
            axes[2, i].set_title(f'Sagittal {slice_idx}')
            axes[2, i].axis('off')

        plt.tight_layout()
        vis_file = output_path / f'atlas_visualization_{self.atlas_type}.png'
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to: {vis_file}")

    def print_label_summary(self, max_labels=20):
        """Print a summary of atlas labels."""
        print(f"\n{'='*70}")
        print(f"Atlas Label Summary ({self.atlas_type.upper()})")
        print(f"{'='*70}")
        print(f"Total regions: {len(self.label_names)}")
        print(f"\nFirst {max_labels} labels:")
        for i, (label_id, label_name) in enumerate(sorted(self.label_names.items())[:max_labels]):
            print(f"  {label_id:3d}: {label_name}")
        if len(self.label_names) > max_labels:
            print(f"  ... ({len(self.label_names) - max_labels} more regions)")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess brain atlas for TextBraTS'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Disk1/afrouz/Data/Merged/',
        help='Directory containing BraTS data (to get reference image)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/',
        help='Output directory for preprocessed atlas'
    )
    parser.add_argument(
        '--target_shape',
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help='Target shape for resampled atlas (default: 128 128 128)'
    )
    parser.add_argument(
        '--atlas_type',
        type=str,
        default='harvard-oxford',
        choices=['harvard-oxford', 'aal', 'talairach'],
        help='Type of brain atlas to use'
    )
    parser.add_argument(
        '--use_reference',
        action='store_true',
        help='Use a reference BraTS image for affine/shape (requires --sample_id)'
    )
    parser.add_argument(
        '--sample_id',
        type=str,
        default='BraTS20_Training_001',
        help='Sample ID to use as reference (if --use_reference is set)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate visualizations of the atlas'
    )
    parser.add_argument(
        '--use_padding',
        action='store_true',
        default=False,
        help='Use padding instead of interpolation (preserves exact label IDs)'
    )

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = BrainAtlasPreprocessor(atlas_type=args.atlas_type)

    # Print label summary
    preprocessor.print_label_summary()

    # Resample atlas
    if args.use_padding:
        # Use padding (recommended for discrete labels)
        print("\n⚠️  Using PADDING method (preserves exact labels)")
        resampled_atlas = preprocessor.pad_to_shape(tuple(args.target_shape))
        processing_method = 'padded'
    elif args.use_reference:
        # Use a reference BraTS image
        print(f"\nUsing reference image from sample: {args.sample_id}")
        sample_dir = Path(args.data_dir) / args.sample_id
        flair_path = sample_dir / f"{args.sample_id}_flair.nii.gz"

        if not flair_path.exists():
            print(f"Error: Reference image not found at {flair_path}")
            print("Falling back to target shape method...")
            resampled_atlas = preprocessor.resample_to_shape(tuple(args.target_shape))
            processing_method = 'resampled'
        else:
            reference_img = nib.load(str(flair_path))
            print(f"Reference image shape: {reference_img.shape}")
            print(f"Reference image affine:\n{reference_img.affine}")
            resampled_atlas = preprocessor.resample_to_target(reference_img)
            processing_method = 'resampled'
    else:
        # Use target shape only (with interpolation)
        print("\n⚠️  Using INTERPOLATION method (may blur labels)")
        resampled_atlas = preprocessor.resample_to_shape(tuple(args.target_shape))
        processing_method = 'resampled'

    # Save atlas
    atlas_file, labels_file, stats_file = preprocessor.save_atlas(
        args.output_dir,
        resampled_atlas,
        method=processing_method
    )

    # Visualize
    # if args.visualize:
    preprocessor.visualize_atlas(resampled_atlas, args.output_dir)

    print("\n" + "="*70)
    print("Atlas preprocessing complete!")
    print("="*70)
    print(f"Atlas NIfTI: {atlas_file}")
    print(f"Label mapping: {labels_file}")
    print(f"Statistics: {stats_file}")
    print("\nYou can now use this atlas to create region-specific masks")
    print("based on the anatomical locations in volumetric_extractions.json")


if __name__ == '__main__':
    main()
