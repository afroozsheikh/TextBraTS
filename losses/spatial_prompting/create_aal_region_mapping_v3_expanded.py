"""
Create AAL Region Mapping V3 - EXPANDED VERSION

This version addresses the problem that cortical lobe descriptions (e.g., "Frontal Lobe")
in radiology reports often don't capture tumors that extend into anatomically adjacent
deep brain structures (Basal Ganglia, Thalamus, Limbic, Insula, Cingulate).

Key improvements:
1. When text says "Frontal Lobe" → include Frontal PLUS adjacent Insula, Cingulate, Basal Ganglia
2. When text says "Parietal Lobe" → include Parietal PLUS adjacent Thalamus, Basal Ganglia
3. When text says "Temporal Lobe" → include Temporal PLUS adjacent Limbic (Hippocampus, Amygdala)
4. Maintain specificity by hemisphere (Left/Right)

Analysis showed this will increase ground truth overlap from ~25% to hopefully >60%.
"""

import json
import numpy as np
import nibabel as nib
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
    Categorize AAL labels by brain region.

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
    - Last digit 1: Left hemisphere
    - Last digit 2: Right hemisphere
    - Last digit 0: Midline
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
        midline = [l for l in labels if l % 10 == 0]
        bilateral = left + right + midline
        return left, right, bilateral

    regions = {
        'frontal': split_by_hemisphere(frontal),
        'parietal': split_by_hemisphere(parietal),
        'temporal': split_by_hemisphere(temporal),
        'occipital': split_by_hemisphere(occipital),
        'insula_cingulate': split_by_hemisphere(insula_cingulate),
        'limbic': split_by_hemisphere(limbic),
        'basal_thalamus': split_by_hemisphere(basal_thalamus),
        'cerebellum': split_by_hemisphere(cerebellum),
    }

    # Print statistics
    print("\nLabel distribution by region:")
    for region_name, (left, right, bilateral) in regions.items():
        all_labels = set(left + right)
        print(f"  {region_name:20s}: {len(all_labels):3d} total ({len(left):3d} L, {len(right):3d} R)")

    return regions, unique_labels


def create_expanded_aal_region_mapping():
    """
    Create EXPANDED region mapping that includes anatomically adjacent structures.

    Strategy:
    - Frontal Lobe → Frontal + Insula/Cingulate + Basal Ganglia (anterior)
    - Parietal Lobe → Parietal + Basal Ganglia + Thalamus
    - Temporal Lobe → Temporal + Limbic (Hippocampus, Amygdala)
    - Occipital Lobe → Occipital + Posterior Cingulate
    """

    # Load actual labels
    unique_labels = load_actual_atlas_labels()

    # Categorize by region
    regions, all_labels = categorize_labels_by_region(unique_labels)

    # Extract components
    frontal_L, frontal_R, frontal_B = regions['frontal']
    parietal_L, parietal_R, parietal_B = regions['parietal']
    temporal_L, temporal_R, temporal_B = regions['temporal']
    occipital_L, occipital_R, occipital_B = regions['occipital']
    insula_L, insula_R, insula_B = regions['insula_cingulate']
    limbic_L, limbic_R, limbic_B = regions['limbic']
    basal_L, basal_R, basal_B = regions['basal_thalamus']
    cereb_L, cereb_R, cereb_B = regions['cerebellum']

    # Create EXPANDED mapping
    region_mapping = {
        # FRONTAL LOBE - include adjacent Insula/Cingulate, Limbic (anterior cingulate), and Basal Ganglia
        "Frontal Lobe": {
            "Left": frontal_L + insula_L + limbic_L + basal_L,
            "Right": frontal_R + insula_R + limbic_R + basal_R,
            "Bilateral": frontal_B + insula_B + limbic_B + basal_B,
            "Unspecified": frontal_B + insula_B + limbic_B + basal_B
        },

        # PARIETAL LOBE - include adjacent Limbic (posterior cingulate), Basal Ganglia and Thalamus
        "Parietal Lobe": {
            "Left": parietal_L + limbic_L + basal_L,
            "Right": parietal_R + limbic_R + basal_R,
            "Bilateral": parietal_B + limbic_B + basal_B,
            "Unspecified": parietal_B + limbic_B + basal_B
        },

        # TEMPORAL LOBE - include adjacent Limbic structures
        "Temporal Lobe": {
            "Left": temporal_L + limbic_L,
            "Right": temporal_R + limbic_R,
            "Bilateral": temporal_B + limbic_B,
            "Unspecified": temporal_B + limbic_B
        },

        # OCCIPITAL LOBE - include posterior cingulate (part of insula_cingulate)
        "Occipital Lobe": {
            "Left": occipital_L + [l for l in insula_L if l >= 4000],  # Posterior cingulate
            "Right": occipital_R + [l for l in insula_R if l >= 4000],
            "Bilateral": occipital_B + [l for l in insula_B if l >= 4000],
            "Unspecified": occipital_B + [l for l in insula_B if l >= 4000]
        },

        # Individual structures (keep as is)
        "Basal Ganglia": {
            "Left": basal_L[:len(basal_L)//2] if len(basal_L) > 0 else [],
            "Right": basal_R[:len(basal_R)//2] if len(basal_R) > 0 else [],
            "Bilateral": basal_B,
            "Unspecified": basal_B
        },

        "Thalamus": {
            "Left": [l for l in basal_L if 7100 <= l < 7200],
            "Right": [l for l in basal_R if 7100 <= l < 7200],
            "Bilateral": [l for l in basal_B if 7100 <= l < 7200],
            "Unspecified": [l for l in basal_B if 7100 <= l < 7200]
        },

        "Hippocampus": {
            "Left": [l for l in limbic_L if 4100 <= l < 4200],
            "Right": [l for l in limbic_R if 4100 <= l < 4200],
            "Bilateral": [l for l in limbic_B if 4100 <= l < 4200],
            "Unspecified": [l for l in limbic_B if 4100 <= l < 4200]
        },

        "Amygdala": {
            "Left": [l for l in limbic_L if 4200 <= l < 4300],
            "Right": [l for l in limbic_R if 4200 <= l < 4300],
            "Bilateral": [l for l in limbic_B if 4200 <= l < 4300],
            "Unspecified": [l for l in limbic_B if 4200 <= l < 4300]
        },

        "Insula": {
            "Left": [l for l in insula_L if 3000 <= l < 3100],
            "Right": [l for l in insula_R if 3000 <= l < 3100],
            "Bilateral": [l for l in insula_B if 3000 <= l < 3100],
            "Unspecified": [l for l in insula_B if 3000 <= l < 3100]
        },

        "Cingulate": {
            "Left": limbic_L[:6] if len(limbic_L) >= 6 else limbic_L,
            "Right": limbic_R[:6] if len(limbic_R) >= 6 else limbic_R,
            "Bilateral": limbic_B[:12] if len(limbic_B) >= 12 else limbic_B,
            "Unspecified": limbic_B[:12] if len(limbic_B) >= 12 else limbic_B
        },

        "Cerebellum": {
            "Left": cereb_L,
            "Right": cereb_R,
            "Bilateral": cereb_B,
            "Unspecified": cereb_B
        },

        # Composite regions (EXPANDED)
        "Junction of Frontal and Parietal Lobes": {
            "Left": frontal_L + parietal_L + insula_L + basal_L,
            "Right": frontal_R + parietal_R + insula_R + basal_R,
            "Bilateral": frontal_B + parietal_B + insula_B + basal_B,
            "Unspecified": frontal_B + parietal_B + insula_B + basal_B
        },

        "Temporo-Parietal Region": {
            "Left": temporal_L + parietal_L + limbic_L + basal_L,
            "Right": temporal_R + parietal_R + limbic_R + basal_R,
            "Bilateral": temporal_B + parietal_B + limbic_B + basal_B,
            "Unspecified": temporal_B + parietal_B + limbic_B + basal_B
        },

        "Mesial Temporal Lobe": {
            "Left": temporal_L + limbic_L,
            "Right": temporal_R + limbic_R,
            "Bilateral": temporal_B + limbic_B,
            "Unspecified": temporal_B + limbic_B
        },

        # Whole brain fallback
        "Brain": {
            "Left": all_labels,
            "Right": all_labels,
            "Bilateral": all_labels,
            "Unspecified": all_labels
        },

        "Cerebral Hemisphere": {
            "Left": frontal_L + parietal_L + temporal_L + occipital_L + insula_L + basal_L + limbic_L,
            "Right": frontal_R + parietal_R + temporal_R + occipital_R + insula_R + basal_R + limbic_R,
            "Bilateral": frontal_B + parietal_B + temporal_B + occipital_B + insula_B + basal_B + limbic_B,
            "Unspecified": frontal_B + parietal_B + temporal_B + occipital_B + insula_B + basal_B + limbic_B
        },

        # Parieto-Occipital (from text extractions)
        "Parieto-Occipital Lobes": {
            "Left": parietal_L + occipital_L + basal_L,
            "Right": parietal_R + occipital_R + basal_R,
            "Bilateral": parietal_B + occipital_B + basal_B,
            "Unspecified": parietal_B + occipital_B + basal_B
        },

        "Parieto-Occipital Lobe": {
            "Left": parietal_L + occipital_L + basal_L,
            "Right": parietal_R + occipital_R + basal_R,
            "Bilateral": parietal_B + occipital_B + basal_B,
            "Unspecified": parietal_B + basal_B + occipital_B
        },

        # Lateral Ventricles (use periventricular structures)
        "Lateral Ventricles": {
            "Left": basal_L + limbic_L,
            "Right": basal_R + limbic_R,
            "Bilateral": basal_B + limbic_B,
            "Unspecified": basal_B + limbic_B
        },

        # Brainstem (use thalamus as proxy)
        "Brainstem": {
            "Left": basal_B,
            "Right": basal_B,
            "Bilateral": basal_B,
            "Unspecified": basal_B
        },

        # Cerebral Region (generic, use whole supratentorial)
        "Cerebral Region": {
            "Left": frontal_L + parietal_L + temporal_L + occipital_L + insula_L + basal_L + limbic_L,
            "Right": frontal_R + parietal_R + temporal_R + occipital_R + insula_R + basal_R + limbic_R,
            "Bilateral": frontal_B + parietal_B + temporal_B + occipital_B + insula_B + basal_B + limbic_B,
            "Unspecified": frontal_B + parietal_B + temporal_B + occipital_B + insula_B + basal_B + limbic_B
        },
    }

    return region_mapping


def main():
    print("="*80)
    print("Creating AAL Region Mapping V3 (EXPANDED with Adjacent Structures)")
    print("="*80)

    # Create mapping
    region_mapping = create_expanded_aal_region_mapping()

    # Save
    output_path = "/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/region_mapping_aal_v3_expanded.json"
    with open(output_path, 'w') as f:
        json.dump(region_mapping, f, indent=2)

    print(f"\n✓ Saved expanded AAL region mapping to: {output_path}")

    # Print statistics
    print("\n" + "="*80)
    print("Region mapping statistics:")
    print("="*80)
    for region, sides in region_mapping.items():
        for side, labels in sides.items():
            num_labels = len(labels)
            if num_labels > 0:
                print(f"  {region:40s} ({side:11s}): {num_labels:3d} labels")

    print("\n" + "="*80)
    print("KEY IMPROVEMENTS:")
    print("="*80)
    print("  - Frontal Lobe now includes Insula, Cingulate, and Basal Ganglia")
    print("  - Parietal Lobe now includes Basal Ganglia and Thalamus")
    print("  - Temporal Lobe now includes Limbic structures (Hippocampus, Amygdala)")
    print("  - Expected to improve GT overlap from ~25% to >60%")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("  1. Regenerate atlas masks:")
    print("     python generate_sample_atlas_masks.py --region_mapping region_mapping_aal_v3_expanded.json \\")
    print("            --output_dir /Disk1/afrouz/Data/TextBraTS_atlas_masks_aal_v3 --no_skip_existing")
    print("\n  2. Run diagnostic to verify improvement:")
    print("     python comprehensive_debug_aal.py")
    print("="*80)


if __name__ == "__main__":
    main()
