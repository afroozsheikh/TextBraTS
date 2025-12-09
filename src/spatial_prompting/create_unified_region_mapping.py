#!/usr/bin/env python3
"""
Unified Region Mapping Generator for TextBraTS Spatial Prompting

This script creates region mappings for ALL supported atlas types:
- Harvard-Oxford (cortical + subcortical)
- AAL (Automated Anatomical Labeling)
- Talairach (lobe-level)

Maps anatomical region names (from radiology reports) to atlas label IDs.

Usage:
    # Generate all atlases
    python create_unified_region_mapping.py --atlas all

    # Generate specific atlas
    python create_unified_region_mapping.py --atlas harvard-oxford
    python create_unified_region_mapping.py --atlas aal
    python create_unified_region_mapping.py --atlas talairach

    # Use expanded mode (includes adjacent structures for better coverage)
    python create_unified_region_mapping.py --atlas aal --expanded

Output:
    - region_mapping_<atlas>.json
    - region_mapping_<atlas>_expanded.json (if --expanded)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class UnifiedRegionMapper:
    """Unified region mapping generator for all atlas types."""

    def __init__(self, atlas_type: str, expanded: bool = False):
        """
        Initialize mapper.

        Args:
            atlas_type: 'harvard-oxford', 'aal', or 'talairach'
            expanded: If True, include anatomically adjacent structures
        """
        self.atlas_type = atlas_type
        self.expanded = expanded
        self.region_mapping = {}

        print(f"\n{'='*80}")
        print(f"CREATING {atlas_type.upper()} REGION MAPPING")
        if expanded:
            print("MODE: EXPANDED (includes adjacent anatomical structures)")
        else:
            print("MODE: STANDARD (lobe-specific only)")
        print(f"{'='*80}\n")

    def generate_mapping(self) -> Dict:
        """Generate region mapping based on atlas type."""
        if self.atlas_type == 'harvard-oxford':
            return self._create_harvard_oxford_mapping()
        elif self.atlas_type == 'aal':
            return self._create_aal_mapping()
        elif self.atlas_type == 'talairach':
            return self._create_talairach_mapping()
        else:
            raise ValueError(f"Unsupported atlas type: {self.atlas_type}")

    # ============================================================================
    # HARVARD-OXFORD ATLAS
    # ============================================================================

    def _create_harvard_oxford_mapping(self) -> Dict:
        """Create Harvard-Oxford atlas mapping."""

        # Cortical lobes (1-48 are BILATERAL/SYMMETRIC in H-O)
        frontal_cortical = [1, 3, 4, 5, 6, 7, 25, 26, 33, 41]
        parietal_cortical = [17, 18, 19, 20, 21, 31, 43]
        temporal_cortical = [8, 9, 10, 11, 12, 13, 14, 15, 16, 37, 38, 44, 45, 46]
        occipital_cortical = [22, 23, 24, 32, 36, 39, 40, 47, 48]
        insula = [2, 42]
        limbic_cingulate = [27, 28, 29, 30, 34, 35]

        # Subcortical (49-69 are LATERALIZED)
        subcortical_left = [52, 53, 54, 55, 57, 58]  # Thalamus, BG, Hippocampus, Amygdala
        subcortical_right = [63, 64, 65, 66, 68, 69]
        brainstem = [49]  # Brain-Stem (midline)

        # Expanded mode: include adjacent structures
        if self.expanded:
            frontal_expanded_left = frontal_cortical + insula + [29] + subcortical_left[:4]
            frontal_expanded_right = frontal_cortical + insula + [29] + subcortical_right[:4]
            parietal_expanded_left = parietal_cortical + [30] + subcortical_left[:2]
            parietal_expanded_right = parietal_cortical + [30] + subcortical_right[:2]
            temporal_expanded_left = temporal_cortical + [34, 35] + subcortical_left[4:]
            temporal_expanded_right = temporal_cortical + [34, 35] + subcortical_right[4:]
        else:
            frontal_expanded_left = frontal_expanded_right = frontal_cortical
            parietal_expanded_left = parietal_expanded_right = parietal_cortical
            temporal_expanded_left = temporal_expanded_right = temporal_cortical

        mapping = {}

        # Cerebral Lobes
        mapping["Frontal Lobe"] = {
            "Left": frontal_expanded_left,
            "Right": frontal_expanded_right,
            "Bilateral": frontal_expanded_left + frontal_expanded_right,
            "Unspecified": frontal_expanded_left + frontal_expanded_right
        }

        mapping["Parietal Lobe"] = {
            "Left": parietal_expanded_left,
            "Right": parietal_expanded_right,
            "Bilateral": parietal_expanded_left + parietal_expanded_right,
            "Unspecified": parietal_expanded_left + parietal_expanded_right
        }

        mapping["Temporal Lobe"] = {
            "Left": temporal_expanded_left,
            "Right": temporal_expanded_right,
            "Bilateral": temporal_expanded_left + temporal_expanded_right,
            "Unspecified": temporal_expanded_left + temporal_expanded_right
        }

        mapping["Occipital Lobe"] = {
            "Left": occipital_cortical,
            "Right": occipital_cortical,
            "Bilateral": occipital_cortical,
            "Unspecified": occipital_cortical
        }

        # Hemispheres
        left_hemi = frontal_expanded_left + parietal_expanded_left + temporal_expanded_left + occipital_cortical + subcortical_left
        right_hemi = frontal_expanded_right + parietal_expanded_right + temporal_expanded_right + occipital_cortical + subcortical_right

        mapping["Cerebral Hemisphere"] = {
            "Left": left_hemi,
            "Right": right_hemi,
            "Bilateral": left_hemi + right_hemi,
            "Unspecified": left_hemi + right_hemi
        }

        # Basal Ganglia
        mapping["Basal Ganglia"] = {
            "Left": [53, 54, 55],  # Caudate, Putamen, Pallidum
            "Right": [64, 65, 66],
            "Bilateral": [53, 54, 55, 64, 65, 66],
            "Unspecified": [53, 54, 55, 64, 65, 66]
        }

        # Brainstem
        mapping["Brainstem"] = {
            "Left": [49],
            "Right": [49],
            "Bilateral": [49],
            "Unspecified": [49]
        }

        # Ventricles (use adjacent structures as proxy)
        periventricular_left = subcortical_left + frontal_cortical + parietal_cortical
        periventricular_right = subcortical_right + frontal_cortical + parietal_cortical

        mapping["Lateral Ventricle"] = {
            "Left": periventricular_left,
            "Right": periventricular_right,
            "Bilateral": periventricular_left + periventricular_right,
            "Unspecified": periventricular_left + periventricular_right
        }

        # Aliases
        mapping["Lateral Ventricles"] = mapping["Lateral Ventricle"]
        mapping["Ventricle"] = mapping["Lateral Ventricle"]
        mapping["Ventricles"] = mapping["Lateral Ventricle"]

        # Junctions
        mapping["Frontal-Parietal Junction"] = {
            "Left": frontal_expanded_left + parietal_expanded_left,
            "Right": frontal_expanded_right + parietal_expanded_right,
            "Bilateral": frontal_expanded_left + parietal_expanded_left + frontal_expanded_right + parietal_expanded_right,
            "Unspecified": frontal_expanded_left + parietal_expanded_left + frontal_expanded_right + parietal_expanded_right
        }

        mapping["Temporal-Parietal Junction"] = {
            "Left": temporal_expanded_left + parietal_expanded_left,
            "Right": temporal_expanded_right + parietal_expanded_right,
            "Bilateral": temporal_expanded_left + parietal_expanded_left + temporal_expanded_right + parietal_expanded_right,
            "Unspecified": temporal_expanded_left + parietal_expanded_left + temporal_expanded_right + parietal_expanded_right
        }

        # Whole brain fallback
        whole_brain = left_hemi + right_hemi + brainstem
        mapping["Brain"] = {
            "Left": whole_brain,
            "Right": whole_brain,
            "Bilateral": whole_brain,
            "Unspecified": whole_brain
        }

        # Vague descriptors
        for vague in ["Lesion Center", "Lesion Core", "Lesion Area", "Lesion Areas", "Unspecified"]:
            mapping[vague] = mapping["Brain"]

        return mapping

    # ============================================================================
    # AAL ATLAS
    # ============================================================================

    def _create_aal_mapping(self) -> Dict:
        """
        Create AAL atlas mapping.

        AAL label structure:
        - 2xxx: Frontal lobe
        - 3xxx: Insula + Cingulate
        - 4xxx: Limbic (Hippocampus, Amygdala)
        - 5xxx: Occipital lobe
        - 6xxx: Parietal lobe
        - 7xxx: Basal ganglia + Thalamus
        - 8xxx: Temporal lobe
        - 9xxx: Cerebellum

        Laterality (last digit):
        - 1: Left hemisphere
        - 2: Right hemisphere
        - 0: Midline
        """

        # Try to load actual atlas labels if available
        try:
            atlas_path = "/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/brain_atlas_aal_resampled.nii.gz"
            import nibabel as nib
            atlas = nib.load(atlas_path)
            atlas_data = atlas.get_fdata()
            unique_labels = [int(x) for x in np.unique(atlas_data) if x > 0]
            print(f"✓ Loaded {len(unique_labels)} unique labels from AAL atlas")
        except:
            print("⚠ Could not load AAL atlas, using predefined labels")
            # Predefined labels (covers most common AAL regions)
            unique_labels = list(range(2001, 2999)) + list(range(3001, 3999)) + \
                          list(range(4001, 4999)) + list(range(5001, 5999)) + \
                          list(range(6001, 6999)) + list(range(7001, 7999)) + \
                          list(range(8001, 8999))

        # Categorize by region and hemisphere
        def get_region_labels(region_code, unique_labels):
            region = [l for l in unique_labels if region_code <= l < region_code + 1000]
            left = [l for l in region if l % 10 == 1]
            right = [l for l in region if l % 10 == 2]
            midline = [l for l in region if l % 10 == 0]
            bilateral = left + right + midline
            return left, right, bilateral

        frontal_l, frontal_r, frontal_b = get_region_labels(2000, unique_labels)
        insula_cingulate_l, insula_cingulate_r, insula_cingulate_b = get_region_labels(3000, unique_labels)
        limbic_l, limbic_r, limbic_b = get_region_labels(4000, unique_labels)
        occipital_l, occipital_r, occipital_b = get_region_labels(5000, unique_labels)
        parietal_l, parietal_r, parietal_b = get_region_labels(6000, unique_labels)
        basal_thalamus_l, basal_thalamus_r, basal_thalamus_b = get_region_labels(7000, unique_labels)
        temporal_l, temporal_r, temporal_b = get_region_labels(8000, unique_labels)

        # Expanded mode: include adjacent structures
        if self.expanded:
            frontal_exp_l = frontal_l + insula_cingulate_l + basal_thalamus_l
            frontal_exp_r = frontal_r + insula_cingulate_r + basal_thalamus_r
            parietal_exp_l = parietal_l + basal_thalamus_l
            parietal_exp_r = parietal_r + basal_thalamus_r
            temporal_exp_l = temporal_l + limbic_l
            temporal_exp_r = temporal_r + limbic_r
        else:
            frontal_exp_l, frontal_exp_r = frontal_l, frontal_r
            parietal_exp_l, parietal_exp_r = parietal_l, parietal_r
            temporal_exp_l, temporal_exp_r = temporal_l, temporal_r

        mapping = {}

        # Cerebral Lobes
        mapping["Frontal Lobe"] = {
            "Left": frontal_exp_l,
            "Right": frontal_exp_r,
            "Bilateral": frontal_exp_l + frontal_exp_r,
            "Unspecified": frontal_exp_l + frontal_exp_r
        }

        mapping["Parietal Lobe"] = {
            "Left": parietal_exp_l,
            "Right": parietal_exp_r,
            "Bilateral": parietal_exp_l + parietal_exp_r,
            "Unspecified": parietal_exp_l + parietal_exp_r
        }

        mapping["Temporal Lobe"] = {
            "Left": temporal_exp_l,
            "Right": temporal_exp_r,
            "Bilateral": temporal_exp_l + temporal_exp_r,
            "Unspecified": temporal_exp_l + temporal_exp_r
        }

        mapping["Occipital Lobe"] = {
            "Left": occipital_l,
            "Right": occipital_r,
            "Bilateral": occipital_b,
            "Unspecified": occipital_b
        }

        # Basal Ganglia
        mapping["Basal Ganglia"] = {
            "Left": basal_thalamus_l,
            "Right": basal_thalamus_r,
            "Bilateral": basal_thalamus_b,
            "Unspecified": basal_thalamus_b
        }

        # Ventricles (use adjacent structures)
        periventricular_l = basal_thalamus_l + frontal_l + parietal_l + temporal_l
        periventricular_r = basal_thalamus_r + frontal_r + parietal_r + temporal_r

        mapping["Lateral Ventricle"] = {
            "Left": periventricular_l,
            "Right": periventricular_r,
            "Bilateral": periventricular_l + periventricular_r,
            "Unspecified": periventricular_l + periventricular_r
        }

        mapping["Lateral Ventricles"] = mapping["Lateral Ventricle"]
        mapping["Ventricle"] = mapping["Lateral Ventricle"]
        mapping["Ventricles"] = mapping["Lateral Ventricle"]

        # Whole brain
        whole_brain = frontal_b + parietal_b + temporal_b + occipital_b + basal_thalamus_b + insula_cingulate_b + limbic_b
        mapping["Brain"] = {
            "Left": whole_brain,
            "Right": whole_brain,
            "Bilateral": whole_brain,
            "Unspecified": whole_brain
        }

        # Vague descriptors
        for vague in ["Lesion Center", "Lesion Core", "Lesion Area", "Lesion Areas", "Unspecified"]:
            mapping[vague] = mapping["Brain"]

        return mapping

    # ============================================================================
    # TALAIRACH ATLAS
    # ============================================================================

    def _create_talairach_mapping(self) -> Dict:
        """
        Create Talairach atlas mapping.

        Talairach lobe-level labels (approximate):
        1: Left Frontal
        2: Left Parietal
        3: Left Temporal
        4: Left Occipital
        5: Left Limbic
        6: Left SubLobar (deep structures)
        7: Right Frontal
        8: Right Parietal
        9: Right Temporal
        10: Right Occipital
        11: Right Limbic
        12: Right SubLobar
        13: Brainstem
        14: Left Cerebellum
        15: Right Cerebellum
        """

        tal_labels = {
            "Left_Frontal": 1,
            "Left_Parietal": 2,
            "Left_Temporal": 3,
            "Left_Occipital": 4,
            "Left_Limbic": 5,
            "Left_SubLobar": 6,
            "Right_Frontal": 7,
            "Right_Parietal": 8,
            "Right_Temporal": 9,
            "Right_Occipital": 10,
            "Right_Limbic": 11,
            "Right_SubLobar": 12,
            "Brainstem": 13,
            "Cerebellum_Left": 14,
            "Cerebellum_Right": 15,
        }

        left_hemi = [1, 2, 3, 4, 5, 6]
        right_hemi = [7, 8, 9, 10, 11, 12]

        # Expanded mode: include adjacent SubLobar for lobes
        if self.expanded:
            frontal_l_exp = [1, 6]  # Frontal + SubLobar
            frontal_r_exp = [7, 12]
            parietal_l_exp = [2, 6]
            parietal_r_exp = [8, 12]
            temporal_l_exp = [3, 5, 6]  # Temporal + Limbic + SubLobar
            temporal_r_exp = [9, 11, 12]
        else:
            frontal_l_exp, frontal_r_exp = [1], [7]
            parietal_l_exp, parietal_r_exp = [2], [8]
            temporal_l_exp, temporal_r_exp = [3], [9]

        mapping = {}

        # Cerebral Lobes
        mapping["Frontal Lobe"] = {
            "Left": frontal_l_exp,
            "Right": frontal_r_exp,
            "Bilateral": frontal_l_exp + frontal_r_exp,
            "Unspecified": frontal_l_exp + frontal_r_exp
        }

        mapping["Parietal Lobe"] = {
            "Left": parietal_l_exp,
            "Right": parietal_r_exp,
            "Bilateral": parietal_l_exp + parietal_r_exp,
            "Unspecified": parietal_l_exp + parietal_r_exp
        }

        mapping["Temporal Lobe"] = {
            "Left": temporal_l_exp,
            "Right": temporal_r_exp,
            "Bilateral": temporal_l_exp + temporal_r_exp,
            "Unspecified": temporal_l_exp + temporal_r_exp
        }

        mapping["Occipital Lobe"] = {
            "Left": [4],
            "Right": [10],
            "Bilateral": [4, 10],
            "Unspecified": [4, 10]
        }

        # Hemispheres
        mapping["Cerebral Hemisphere"] = {
            "Left": left_hemi,
            "Right": right_hemi,
            "Bilateral": left_hemi + right_hemi,
            "Unspecified": left_hemi + right_hemi
        }

        # Basal Ganglia (SubLobar regions)
        mapping["Basal Ganglia"] = {
            "Left": [6],
            "Right": [12],
            "Bilateral": [6, 12],
            "Unspecified": [6, 12]
        }

        # Brainstem
        mapping["Brainstem"] = {
            "Left": [13],
            "Right": [13],
            "Bilateral": [13],
            "Unspecified": [13]
        }

        # Ventricles (use SubLobar + adjacent lobes as proxy)
        periventricular_l = [6, 1, 2, 3, 4]
        periventricular_r = [12, 7, 8, 9, 10]

        mapping["Lateral Ventricle"] = {
            "Left": periventricular_l,
            "Right": periventricular_r,
            "Bilateral": periventricular_l + periventricular_r,
            "Unspecified": periventricular_l + periventricular_r
        }

        mapping["Lateral Ventricles"] = mapping["Lateral Ventricle"]
        mapping["Ventricle"] = mapping["Lateral Ventricle"]
        mapping["Ventricles"] = mapping["Lateral Ventricle"]

        # Whole brain
        whole_brain = left_hemi + right_hemi + [13, 14, 15]
        mapping["Brain"] = {
            "Left": whole_brain,
            "Right": whole_brain,
            "Bilateral": whole_brain,
            "Unspecified": whole_brain
        }

        # Vague descriptors
        for vague in ["Lesion Center", "Lesion Core", "Lesion Area", "Lesion Areas", "Unspecified"]:
            mapping[vague] = mapping["Brain"]

        return mapping

    # ============================================================================
    # UTILITIES
    # ============================================================================

    def save_mapping(self, mapping: Dict, output_path: Path):
        """Save mapping to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"✓ Saved region mapping to: {output_path}")

    def print_statistics(self, mapping: Dict):
        """Print mapping statistics."""
        print(f"\n{'='*80}")
        print(f"MAPPING STATISTICS - {self.atlas_type.upper()}")
        print(f"{'='*80}\n")

        print(f"Total mapped regions: {len(mapping)}")

        # Categorize
        lobes = [k for k in mapping.keys() if "Lobe" in k]
        ventricles = [k for k in mapping.keys() if "Ventricle" in k or "Ventricular" in k]
        subcortical = [k for k in mapping.keys() if k in ["Basal Ganglia", "Brainstem"]]

        print(f"\nRegion categories:")
        print(f"  Lobes: {len(lobes)}")
        print(f"  Ventricles: {len(ventricles)}")
        print(f"  Subcortical: {len(subcortical)}")

        # Sample mapping
        print(f"\nSample mappings:")
        for region in ["Frontal Lobe", "Lateral Ventricle", "Basal Ganglia"]:
            if region in mapping:
                print(f"\n{region}:")
                for side in ["Left", "Right", "Bilateral"]:
                    labels = mapping[region][side]
                    print(f"  {side:10s}: {len(labels)} labels")


def main():
    parser = argparse.ArgumentParser(
        description='Create unified region mappings for TextBraTS spatial prompting'
    )
    parser.add_argument(
        '--atlas',
        type=str,
        choices=['harvard-oxford', 'aal', 'talairach', 'all'],
        default='all',
        help='Atlas type to generate mapping for'
    )
    parser.add_argument(
        '--expanded',
        action='store_true',
        help='Include anatomically adjacent structures for better coverage'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for region mappings'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which atlases to process
    if args.atlas == 'all':
        atlas_types = ['harvard-oxford', 'aal', 'talairach']
    else:
        atlas_types = [args.atlas]

    # Generate mappings
    for atlas_type in atlas_types:
        mapper = UnifiedRegionMapper(atlas_type, expanded=args.expanded)
        mapping = mapper.generate_mapping()

        # Save
        suffix = "_expanded" if args.expanded else ""
        output_path = output_dir / f"region_mapping_{atlas_type}{suffix}.json"
        mapper.save_mapping(mapping, output_path)

        # Statistics
        mapper.print_statistics(mapping)

    # Final summary
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}\n")

    print("Generated mappings:")
    for atlas_type in atlas_types:
        suffix = "_expanded" if args.expanded else ""
        path = output_dir / f"region_mapping_{atlas_type}{suffix}.json"
        print(f"  ✓ {path}")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}\n")
    print("1. Download and resample atlas:")
    print("   python generate_brain_atlas_masks.py --atlas_type <atlas>")
    print("\n2. Generate sample-specific masks:")
    print("   python generate_sample_atlas_masks.py --atlas <atlas>")
    print("\n3. Train with spatial prompting:")
    print("   python main.py --spatial_prompting --atlas_masks_dir /path/to/masks")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
