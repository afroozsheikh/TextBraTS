"""
Create AAL Region Mapping for Spatial Prompting - VERSION 2

This version uses ONLY the actual labels that exist in the resampled atlas,
not broad ranges.
"""

import json
import nibabel as nib
import numpy as np
from pathlib import Path


def load_actual_atlas_labels():
    """Load the actual labels present in the resampled AAL atlas."""
    atlas_path = "/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/brain_atlas_aal_resampled.nii.gz"
    atlas = nib.load(atlas_path)
    atlas_data = atlas.get_fdata()

    # Get all unique labels (excluding background 0)
    unique_labels = [int(x) for x in np.unique(atlas_data) if x > 0]

    print(f"Loaded {len(unique_labels)} unique labels from atlas")
    print(f"Label range: {min(unique_labels)} - {max(unique_labels)}")

    return unique_labels


def categorize_labels_by_region(unique_labels):
    """
    Categorize AAL labels by brain region based on their encoding.

    AAL encoding:
    - 2xxx: Frontal lobe
    - 3xxx: Insula + Cingulate
    - 4xxx: Limbic (Hippocampus, Amygdala, etc.)
    - 5xxx: Occipital lobe
    - 6xxx: Parietal lobe
    - 7xxx: Basal ganglia + Thalamus
    - 8xxx: Temporal lobe
    - 9xxx: Cerebellum

    Laterality:
    - Last digit odd (1): Left hemisphere
    - Last digit even (2): Right hemisphere
    """

    frontal = [l for l in unique_labels if 2000 <= l < 3000]
    insula_cingulate = [l for l in unique_labels if 3000 <= l < 4000]
    limbic = [l for l in unique_labels if 4000 <= l < 5000]
    occipital = [l for l in unique_labels if 5000 <= l < 6000]
    parietal = [l for l in unique_labels if 6000 <= l < 7000]
    basal_thalamus = [l for l in unique_labels if 7000 <= l < 8000]
    temporal = [l for l in unique_labels if 8000 <= l < 9000]
    cerebellum = [l for l in unique_labels if 9000 <= l < 10000]

    # Separate by hemisphere
    def split_by_hemisphere(labels):
        left = [l for l in labels if l % 10 == 1]
        right = [l for l in labels if l % 10 == 2]
        # Some labels might end in 0 (midline structures)
        midline = [l for l in labels if l % 10 == 0]
        bilateral = left + right + midline
        return left, right, bilateral

    regions = {
        'frontal': (frontal, *split_by_hemisphere(frontal)),
        'parietal': (parietal, *split_by_hemisphere(parietal)),
        'temporal': (temporal, *split_by_hemisphere(temporal)),
        'occipital': (occipital, *split_by_hemisphere(occipital)),
        'insula': (insula_cingulate[:2] if len(insula_cingulate) >= 2 else insula_cingulate,
                  [3001] if 3001 in insula_cingulate else [],
                  [3002] if 3002 in insula_cingulate else [],
                  [3001, 3002] if len(insula_cingulate) >= 2 else insula_cingulate),
        'cingulate': (limbic[:6] if len(limbic) >= 6 else limbic[:4],
                     [4001, 4011, 4021] if all(l in limbic for l in [4001, 4011, 4021]) else [],
                     [4002, 4012, 4022] if all(l in limbic for l in [4002, 4012, 4022]) else [],
                     limbic[:6] if len(limbic) >= 6 else limbic[:4]),
        'hippocampus': ([4101, 4102] if all(l in limbic for l in [4101, 4102]) else [],
                       [4101] if 4101 in limbic else [],
                       [4102] if 4102 in limbic else [],
                       [4101, 4102] if all(l in limbic for l in [4101, 4102]) else []),
        'amygdala': ([4201, 4202] if all(l in limbic for l in [4201, 4202]) else [],
                    [4201] if 4201 in limbic else [],
                    [4202] if 4202 in limbic else [],
                    [4201, 4202] if all(l in limbic for l in [4201, 4202]) else []),
        'basal_ganglia': (basal_thalamus[:6], *split_by_hemisphere(basal_thalamus[:6])),
        'thalamus': ([7101, 7102] if all(l in basal_thalamus for l in [7101, 7102]) else [],
                    [7101] if 7101 in basal_thalamus else [],
                    [7102] if 7102 in basal_thalamus else [],
                    [7101, 7102] if all(l in basal_thalamus for l in [7101, 7102]) else []),
        'cerebellum': (cerebellum, *split_by_hemisphere(cerebellum)),
    }

    # Print statistics
    print("\nLabel distribution by region:")
    for region_name, (all_labels, left, right, bilateral) in regions.items():
        print(f"  {region_name:15s}: {len(all_labels):3d} total ({len(left):3d} L, {len(right):3d} R)")

    return regions, unique_labels


def create_aal_region_mapping_v2():
    """Create region mapping using ONLY labels that actually exist in the atlas."""

    # Load actual labels
    unique_labels = load_actual_atlas_labels()

    # Categorize by region
    regions, all_labels = categorize_labels_by_region(unique_labels)

    # Create mapping dictionary
    region_mapping = {
        "Frontal Lobe": {
            "Left": regions['frontal'][1],
            "Right": regions['frontal'][2],
            "Bilateral": regions['frontal'][3],
            "Unspecified": regions['frontal'][3]
        },

        "Parietal Lobe": {
            "Left": regions['parietal'][1],
            "Right": regions['parietal'][2],
            "Bilateral": regions['parietal'][3],
            "Unspecified": regions['parietal'][3]
        },

        "Temporal Lobe": {
            "Left": regions['temporal'][1],
            "Right": regions['temporal'][2],
            "Bilateral": regions['temporal'][3],
            "Unspecified": regions['temporal'][3]
        },

        "Occipital Lobe": {
            "Left": regions['occipital'][1],
            "Right": regions['occipital'][2],
            "Bilateral": regions['occipital'][3],
            "Unspecified": regions['occipital'][3]
        },

        "Basal Ganglia": {
            "Left": regions['basal_ganglia'][1],
            "Right": regions['basal_ganglia'][2],
            "Bilateral": regions['basal_ganglia'][3],
            "Unspecified": regions['basal_ganglia'][3]
        },

        "Thalamus": {
            "Left": regions['thalamus'][1],
            "Right": regions['thalamus'][2],
            "Bilateral": regions['thalamus'][3],
            "Unspecified": regions['thalamus'][3]
        },

        "Hippocampus": {
            "Left": regions['hippocampus'][1],
            "Right": regions['hippocampus'][2],
            "Bilateral": regions['hippocampus'][3],
            "Unspecified": regions['hippocampus'][3]
        },

        "Amygdala": {
            "Left": regions['amygdala'][1],
            "Right": regions['amygdala'][2],
            "Bilateral": regions['amygdala'][3],
            "Unspecified": regions['amygdala'][3]
        },

        "Insula": {
            "Left": regions['insula'][1],
            "Right": regions['insula'][2],
            "Bilateral": regions['insula'][3],
            "Unspecified": regions['insula'][3]
        },

        "Cingulate": {
            "Left": regions['cingulate'][1],
            "Right": regions['cingulate'][2],
            "Bilateral": regions['cingulate'][3],
            "Unspecified": regions['cingulate'][3]
        },

        "Brainstem": {
            "Left": regions['thalamus'][3],  # Use thalamus as proxy
            "Right": regions['thalamus'][3],
            "Bilateral": regions['thalamus'][3],
            "Unspecified": regions['thalamus'][3]
        },

        "Cerebellum": {
            "Left": regions['cerebellum'][1],
            "Right": regions['cerebellum'][2],
            "Bilateral": regions['cerebellum'][3],
            "Unspecified": regions['cerebellum'][3]
        },

        # Composite regions
        "Junction of Frontal and Parietal Lobes": {
            "Left": regions['frontal'][1] + regions['parietal'][1],
            "Right": regions['frontal'][2] + regions['parietal'][2],
            "Bilateral": regions['frontal'][3] + regions['parietal'][3],
            "Unspecified": regions['frontal'][3] + regions['parietal'][3]
        },

        "Temporo-Parietal Region": {
            "Left": regions['temporal'][1] + regions['parietal'][1],
            "Right": regions['temporal'][2] + regions['parietal'][2],
            "Bilateral": regions['temporal'][3] + regions['parietal'][3],
            "Unspecified": regions['temporal'][3] + regions['parietal'][3]
        },

        "Mesial Temporal Lobe": {
            "Left": regions['temporal'][1] + regions['hippocampus'][1],
            "Right": regions['temporal'][2] + regions['hippocampus'][2],
            "Bilateral": regions['temporal'][3] + regions['hippocampus'][3],
            "Unspecified": regions['temporal'][3] + regions['hippocampus'][3]
        },

        # Whole brain fallback
        "Brain": {
            "Left": all_labels,
            "Right": all_labels,
            "Bilateral": all_labels,
            "Unspecified": all_labels
        },

        "Cerebral Hemisphere": {
            "Left": regions['frontal'][1] + regions['parietal'][1] + regions['temporal'][1] + regions['occipital'][1],
            "Right": regions['frontal'][2] + regions['parietal'][2] + regions['temporal'][2] + regions['occipital'][2],
            "Bilateral": regions['frontal'][3] + regions['parietal'][3] + regions['temporal'][3] + regions['occipital'][3],
            "Unspecified": regions['frontal'][3] + regions['parietal'][3] + regions['temporal'][3] + regions['occipital'][3]
        },

        # Ventricles (not in AAL, use cortical regions as proxy)
        "Lateral Ventricles": {
            "Left": regions['frontal'][1] + regions['parietal'][1],
            "Right": regions['frontal'][2] + regions['parietal'][2],
            "Bilateral": regions['frontal'][3] + regions['parietal'][3],
            "Unspecified": regions['frontal'][3] + regions['parietal'][3]
        }
    }

    return region_mapping


def main():
    print("="*70)
    print("Creating AAL Region Mapping V2 (Using Only Actual Labels)")
    print("="*70)

    # Create mapping
    region_mapping = create_aal_region_mapping_v2()

    # Save
    output_path = "/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/region_mapping_aal.json"
    with open(output_path, 'w') as f:
        json.dump(region_mapping, f, indent=2)

    print(f"\n✓ Saved AAL region mapping to: {output_path}")

    # Print statistics
    print("\n" + "="*70)
    print("Region mapping statistics:")
    print("="*70)
    for region, sides in region_mapping.items():
        for side, labels in sides.items():
            num_labels = len(labels)
            if num_labels > 0:
                print(f"  {region:35s} ({side:11s}): {num_labels:3d} labels")

    print("\n" + "="*70)
    print("COMPLETE - Ready to regenerate atlas masks!")
    print("="*70)


if __name__ == "__main__":
    main()
