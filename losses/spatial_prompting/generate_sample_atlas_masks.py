"""
Sample-Specific Atlas Mask Generation for TextBraTS Spatial Prompting

This script generates per-sample 3D binary atlas masks that indicate anatomically
plausible regions for each pathology type based on LLM-extracted anatomical annotations
from radiology reports.

Pipeline Step 3 of 4:
1. LLM extraction → volumetric_extractions.json
2. Region mapping → region_mapping.json
3. Atlas mask generation → {sample_id}_atlas_mask.nii.gz (THIS SCRIPT)
4. Network integration → training with spatial prompts

Usage:
    # Process all samples
    python generate_sample_atlas_masks.py

    # Process single sample with visualization
    python generate_sample_atlas_masks.py --sample_id BraTS20_Training_001 --visualize

    # Custom paths
    python generate_sample_atlas_masks.py --atlas_path /path/to/atlas.nii.gz \
                                           --output_dir /path/to/output

Output:
    - Per-sample NIfTI masks: {sample_id}_atlas_mask.nii.gz
    - Shape: (3, 128, 128, 128) - 3 channels for TC, WT, ET
    - Channel 0 (TC): Necrosis regions
    - Channel 1 (WT): Lesion + Edema regions (union)
    - Channel 2 (ET): Lesion regions
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
import os
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SampleAtlasMaskGenerator:
    """
    Generate per-sample atlas masks from LLM-extracted anatomical data.

    Attributes:
        atlas_img: Loaded NIfTI atlas image
        atlas_data: 3D numpy array (128, 128, 128) with integer labels
        affine: Affine transformation matrix from atlas
        region_mapping: Dict mapping region names + sides to atlas label IDs
        volumetric_extractions: Dict of all sample extractions
        output_dir: Path to save generated masks
    """

    def __init__(
        self,
        atlas_path: str,
        region_mapping_path: str,
        volumetric_extractions_path: str,
        output_dir: str
    ):
        """
        Initialize the atlas mask generator.

        Args:
            atlas_path: Path to brain atlas NIfTI file (128x128x128)
            region_mapping_path: Path to region mapping JSON
            volumetric_extractions_path: Path to volumetric extractions JSON
            output_dir: Output directory for generated masks
        """
        print("="*70)
        print("Sample-Specific Atlas Mask Generator")
        print("="*70)

        # Load brain atlas
        print(f"\nLoading brain atlas from: {atlas_path}")
        self.atlas_img = nib.load(atlas_path)
        self.atlas_data = self.atlas_img.get_fdata()
        self.affine = self.atlas_img.affine

        # Validate atlas shape
        assert self.atlas_data.shape == (128, 128, 128), \
            f"Atlas shape must be (128,128,128), got {self.atlas_data.shape}"

        unique_labels = np.unique(self.atlas_data)
        print(f"  ✓ Atlas shape: {self.atlas_data.shape}")
        print(f"  ✓ Unique labels: {len(unique_labels)} (range: {unique_labels.min():.0f}-{unique_labels.max():.0f})")

        # Load region mapping
        print(f"\nLoading region mapping from: {region_mapping_path}")
        with open(region_mapping_path, 'r') as f:
            self.region_mapping = json.load(f)
        print(f"  ✓ Loaded {len(self.region_mapping)} region mappings")

        # Load volumetric extractions
        print(f"\nLoading volumetric extractions from: {volumetric_extractions_path}")
        with open(volumetric_extractions_path, 'r') as f:
            self.volumetric_extractions = json.load(f)
        print(f"  ✓ Loaded {len(self.volumetric_extractions)} sample extractions")

        # Create output directory
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
        print("="*70)

    def _normalize_region_name(self, region: str) -> str:
        """
        Normalize region names for consistent matching.

        Handles:
        - Whitespace variations
        - Plural/singular (Lobes → Lobe)
        - Case variations

        Args:
            region: Raw region name from extractions

        Returns:
            Normalized region name
        """
        normalized = region.strip()
        # Handle plural forms
        normalized = normalized.replace("Lobes", "Lobe")
        normalized = normalized.replace("Regions", "Region")
        normalized = normalized.replace("Areas", "Area")
        return normalized

    def _find_matching_region_key(self, region_name: str) -> Optional[str]:
        """
        Find the best matching region key in region_mapping.

        Strategy:
        1. Exact match (after normalization)
        2. Partial match for variations
        3. Special case mappings for generic terms
        4. Return None if no match found

        Args:
            region_name: Region name to match

        Returns:
            Matching key from region_mapping or None
        """
        normalized = self._normalize_region_name(region_name)

        # Direct match
        if normalized in self.region_mapping:
            return normalized

        # Try singular form
        singular = normalized.rstrip('s')
        if singular in self.region_mapping:
            return singular

        # Special case mappings for generic/ambiguous terms
        special_mappings = {
            'lesion': 'Brain',
            'unspecified': 'Brain',
            'brain tissue': 'Brain',
            'cerebrum': 'Cerebral Hemisphere',
            'deep white matter': 'Brain',
            'white matter': 'Brain',
        }

        if normalized.lower() in special_mappings:
            mapped = special_mappings[normalized.lower()]
            if mapped in self.region_mapping:
                return mapped

        # Partial matching for common variations
        normalized_lower = normalized.lower()
        for key in self.region_mapping.keys():
            key_lower = key.lower()
            # Check if key words match
            if 'frontal' in normalized_lower and 'frontal' in key_lower:
                return key
            if 'parietal' in normalized_lower and 'parietal' in key_lower:
                return key
            if 'temporal' in normalized_lower and 'temporal' in key_lower:
                return key
            if 'occipital' in normalized_lower and 'occipital' in key_lower:
                return key
            if 'basal ganglia' in normalized_lower and 'basal ganglia' in key_lower:
                return key
            if 'thalamus' in normalized_lower and 'thalamus' in key_lower:
                return key

        # No match found
        return None

    def get_label_ids_for_region(self, region_name: str, side: str) -> List[int]:
        """
        Get atlas label IDs for a given anatomical region and laterality.

        Args:
            region_name: Anatomical region (e.g., "Frontal Lobe")
            side: Laterality (Left, Right, Bilateral, Unspecified)

        Returns:
            List of atlas label IDs
        """
        # Find matching region key
        region_key = self._find_matching_region_key(region_name)

        if region_key is None:
            print(f"    [WARNING] Region '{region_name}' not found in mapping, using whole brain")
            # Fallback to whole brain
            if 'Brain' in self.region_mapping and side in self.region_mapping['Brain']:
                return self.region_mapping['Brain'][side]
            else:
                # Ultimate fallback: all labels 1-69
                return list(range(1, 70))

        # Get labels for this region and side
        if region_key in self.region_mapping and side in self.region_mapping[region_key]:
            return self.region_mapping[region_key][side]
        else:
            print(f"    [WARNING] Side '{side}' not found for region '{region_key}', using Unspecified")
            # Try Unspecified as fallback
            if region_key in self.region_mapping and 'Unspecified' in self.region_mapping[region_key]:
                return self.region_mapping[region_key]['Unspecified']
            else:
                # Ultimate fallback
                return list(range(1, 70))

    def create_mask_from_labels(self, label_ids: List[int]) -> np.ndarray:
        """
        Create binary mask from list of atlas label IDs.

        Args:
            label_ids: List of atlas label IDs to include in mask

        Returns:
            Binary mask (128, 128, 128) with float32 dtype
        """
        mask = np.zeros(self.atlas_data.shape, dtype=np.float32)
        for label_id in label_ids:
            mask[self.atlas_data == label_id] = 1.0
        return mask

    def collect_regions_for_pathology(
        self,
        sample_data: dict,
        pathology_type: str
    ) -> List[int]:
        """
        Collect all atlas label IDs for a specific pathology type.

        Args:
            sample_data: Sample data from volumetric_extractions
            pathology_type: Type of pathology (Lesion, Edema, Necrosis)

        Returns:
            Sorted list of unique atlas label IDs
        """
        label_ids = []

        # Get pathology entries
        pathologies = sample_data.get('Pathologies', {})
        entries = pathologies.get(pathology_type, [])

        if not entries:
            return []

        # Collect labels from all entries
        for entry in entries:
            region = entry.get('Region', 'Brain')
            side = entry.get('Side', 'Unspecified')

            # Get label IDs for this region and side
            labels = self.get_label_ids_for_region(region, side)
            label_ids.extend(labels)

        # Return unique sorted list
        return sorted(list(set(label_ids)))

    def generate_channel_masks(self, sample_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3-channel masks (TC, WT, ET) for a sample.

        Channel mapping (BraTS convention):
        - Channel 0 (TC - Tumor Core): Necrosis regions
        - Channel 1 (WT - Whole Tumor): Lesion + Edema regions (union)
        - Channel 2 (ET - Enhancing Tumor): Lesion regions

        Args:
            sample_data: Sample data from volumetric_extractions

        Returns:
            Tuple of (tc_mask, wt_mask, et_mask)
        """
        # Collect label IDs for each pathology type
        necrosis_labels = self.collect_regions_for_pathology(sample_data, 'Necrosis')
        lesion_labels = self.collect_regions_for_pathology(sample_data, 'Lesion')
        edema_labels = self.collect_regions_for_pathology(sample_data, 'Edema')

        # Create masks from labels
        tc_mask = self.create_mask_from_labels(necrosis_labels)
        et_mask = self.create_mask_from_labels(lesion_labels)

        # WT is union of lesion + edema
        wt_labels = sorted(list(set(lesion_labels + edema_labels)))
        wt_mask = self.create_mask_from_labels(wt_labels)

        # Apply whole-brain fallback if needed
        whole_brain_labels = list(range(1, 70))
        whole_brain_mask = self.create_mask_from_labels(whole_brain_labels)

        fallback_applied = []

        if tc_mask.sum() == 0:
            tc_mask = whole_brain_mask.copy()
            fallback_applied.append("TC")

        if wt_mask.sum() == 0:
            wt_mask = whole_brain_mask.copy()
            fallback_applied.append("WT")

        if et_mask.sum() == 0:
            et_mask = whole_brain_mask.copy()
            fallback_applied.append("ET")

        if fallback_applied:
            print(f"  [FALLBACK] {', '.join(fallback_applied)} → Whole brain (no regions found)")

        # Print coverage statistics
        total_voxels = tc_mask.size
        tc_coverage = tc_mask.sum() / total_voxels * 100
        wt_coverage = wt_mask.sum() / total_voxels * 100
        et_coverage = et_mask.sum() / total_voxels * 100
        print(f"  Coverage: TC={tc_coverage:.1f}%, WT={wt_coverage:.1f}%, ET={et_coverage:.1f}%")

        return tc_mask, wt_mask, et_mask

    def save_mask(
        self,
        sample_id: str,
        tc_mask: np.ndarray,
        wt_mask: np.ndarray,
        et_mask: np.ndarray
    ) -> str:
        """
        Save 3-channel mask as NIfTI file.

        Args:
            sample_id: Sample ID (e.g., "BraTS20_Training_001")
            tc_mask: Tumor Core mask
            wt_mask: Whole Tumor mask
            et_mask: Enhancing Tumor mask

        Returns:
            Path to saved file
        """
        # Stack channels (3, 128, 128, 128)
        mask_3d = np.stack([tc_mask, wt_mask, et_mask], axis=0)

        # Create NIfTI image with original atlas affine
        mask_img = nib.Nifti1Image(mask_3d, self.affine)

        # Save
        output_path = os.path.join(self.output_dir, f"{sample_id}_atlas_mask.nii.gz")
        nib.save(mask_img, output_path)

        return output_path

    def visualize_mask(self, sample_id: str, mask_data: np.ndarray) -> str:
        """
        Create visualization of 3-channel mask.

        Args:
            sample_id: Sample ID
            mask_data: Mask data (3, 128, 128, 128)

        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Middle axial slice (z=64)
        slice_idx = 64

        channel_names = ['TC (Tumor Core)', 'WT (Whole Tumor)', 'ET (Enhancing Tumor)']

        for i, (ax, name) in enumerate(zip(axes, channel_names)):
            # Get slice
            slice_data = mask_data[i, :, :, slice_idx].T

            # Display
            im = ax.imshow(slice_data, cmap='hot', origin='lower', vmin=0, vmax=1)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.axis('off')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'Atlas Mask - {sample_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        output_path = os.path.join(self.output_dir, f"{sample_id}_atlas_mask_viz.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def process_sample(self, sample_id: str, visualize: bool = False) -> bool:
        """
        Process a single sample to generate atlas mask.

        Args:
            sample_id: Sample ID to process
            visualize: Whether to generate visualization

        Returns:
            True if successful, False otherwise
        """
        # Check if sample exists
        if sample_id not in self.volumetric_extractions:
            print(f"  ✗ Sample '{sample_id}' not found in volumetric extractions")
            return False

        # Get sample data
        sample_data = self.volumetric_extractions[sample_id]

        # Generate channel masks
        tc_mask, wt_mask, et_mask = self.generate_channel_masks(sample_data)

        # Save mask
        output_path = self.save_mask(sample_id, tc_mask, wt_mask, et_mask)
        print(f"  ✓ Saved: {output_path}")

        # Visualize if requested
        if visualize:
            mask_3d = np.stack([tc_mask, wt_mask, et_mask], axis=0)
            viz_path = self.visualize_mask(sample_id, mask_3d)
            print(f"  ✓ Visualization: {viz_path}")

        return True

    def process_all_samples(
        self,
        visualize: bool = False,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Process all samples in volumetric extractions.

        Args:
            visualize: Whether to generate visualizations
            skip_existing: Skip samples that already have masks

        Returns:
            Statistics dictionary with counts
        """
        total = len(self.volumetric_extractions)
        processed = 0
        skipped = 0
        failed = 0

        print(f"\nProcessing {total} samples...")
        print("-"*70)

        for idx, sample_id in enumerate(sorted(self.volumetric_extractions.keys()), 1):
            print(f"\n[{idx}/{total}] Processing {sample_id}...")

            # Check if already exists
            output_path = os.path.join(self.output_dir, f"{sample_id}_atlas_mask.nii.gz")
            if skip_existing and os.path.exists(output_path):
                print(f"  → Skipped (already exists)")
                skipped += 1
                continue

            # Process sample
            try:
                success = self.process_sample(sample_id, visualize=visualize)
                if success:
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                failed += 1

        return {
            'total': total,
            'processed': processed,
            'skipped': skipped,
            'failed': failed
        }

    def print_statistics(self):
        """Print statistics for existing atlas masks."""
        print("\nAnalyzing existing atlas masks...")
        print("-"*70)

        mask_files = list(Path(self.output_dir).glob("*_atlas_mask.nii.gz"))

        if not mask_files:
            print("No atlas masks found in output directory.")
            return

        stats = {'TC': [], 'WT': [], 'ET': []}

        for mask_file in mask_files:
            try:
                mask = nib.load(str(mask_file)).get_fdata()
                total_voxels = mask[0].size

                stats['TC'].append((mask[0] > 0).sum() / total_voxels * 100)
                stats['WT'].append((mask[1] > 0).sum() / total_voxels * 100)
                stats['ET'].append((mask[2] > 0).sum() / total_voxels * 100)
            except Exception as e:
                print(f"Error loading {mask_file.name}: {e}")

        print(f"\nStatistics for {len(mask_files)} atlas masks:\n")

        for channel in ['TC', 'WT', 'ET']:
            values = stats[channel]
            if values:
                print(f"{channel} Coverage:")
                print(f"  Mean:   {np.mean(values):.2f}%")
                print(f"  Median: {np.median(values):.2f}%")
                print(f"  Min:    {np.min(values):.2f}%")
                print(f"  Max:    {np.max(values):.2f}%")
                print()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate per-sample atlas masks for TextBraTS spatial prompting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all samples
  python generate_sample_atlas_masks.py

  # Process single sample with visualization
  python generate_sample_atlas_masks.py --sample_id BraTS20_Training_001 --visualize

  # Print statistics for existing masks
  python generate_sample_atlas_masks.py --stats
        """
    )

    parser.add_argument(
        "--atlas_path",
        type=str,
        default="/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/brain_atlas_aal_resampled.nii.gz",
        help="Path to resampled brain atlas (128x128x128)"
    )

    parser.add_argument(
        "--region_mapping",
        type=str,
        default="/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/region_mapping_aal.json",
        help="Path to region mapping JSON"
    )

    parser.add_argument(
        "--volumetric_extractions",
        type=str,
        default="/Disk1/afrouz/Projects/TextBraTS/losses/volumetric_extractions.json",
        help="Path to volumetric extractions JSON"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Disk1/afrouz/Data/TextBraTS_atlas_masks_aal",
        help="Output directory for atlas masks"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization PNGs for each mask"
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples that already have masks generated (default: True)"
    )

    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Re-process all samples even if masks exist"
    )

    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="Process only a single sample (for testing)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics for existing masks (don't generate new ones)"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = SampleAtlasMaskGenerator(
        atlas_path=args.atlas_path,
        region_mapping_path=args.region_mapping,
        volumetric_extractions_path=args.volumetric_extractions,
        output_dir=args.output_dir
    )

    # Mode 1: Just print statistics
    if args.stats:
        generator.print_statistics()
        return

    # Mode 2: Process single sample
    if args.sample_id:
        print(f"\nProcessing single sample: {args.sample_id}")
        print("-"*70)
        success = generator.process_sample(args.sample_id, visualize=args.visualize)
        if success:
            print("\n" + "="*70)
            print("SUCCESS")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("FAILED")
            print("="*70)
        return

    # Mode 3: Process all samples
    stats = generator.process_all_samples(
        visualize=args.visualize,
        skip_existing=args.skip_existing
    )

    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total samples:      {stats['total']}")
    print(f"Processed:          {stats['processed']}")
    print(f"Skipped (existing): {stats['skipped']}")
    print(f"Failed:             {stats['failed']}")
    print(f"\nOutput directory: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
