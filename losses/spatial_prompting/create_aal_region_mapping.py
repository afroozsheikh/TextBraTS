"""
Create AAL Region Mapping for Spatial Prompting

Maps anatomical region names (from volumetric_extractions.json) to AAL atlas labels.

The AAL atlas uses encoded labels: XYZZ format where groups indicate brain regions.
This script creates a mapping from clinical anatomical terms to AAL label groups.
"""

import json
from pathlib import Path


def create_aal_region_mapping():
    """
    Create region mapping from anatomical terms to AAL atlas labels.

    AAL atlas label encoding (from the resampled atlas):
    - 2xxx: Frontal lobe regions (Precentral, Frontal_Sup, Frontal_Mid, Frontal_Inf, etc.)
    - 3xxx: Insula and Cingulate (Insula, Cingulum_Ant/Mid/Post)
    - 4xxx: Limbic system (Hippocampus, ParaHippocampal, Amygdala)
    - 5xxx: Occipital lobe (Calcarine, Cuneus, Lingual, Occipital_Sup/Mid/Inf)
    - 6xxx: Parietal lobe (Postcentral, Parietal_Sup/Inf, SupraMarginal, Angular, Precuneus)
    - 7xxx: Basal ganglia (Caudate, Putamen, Pallidum, Thalamus)
    - 8xxx: Temporal lobe (Heschl, Temporal_Sup/Mid/Inf, Temporal_Pole)
    - 9xxx: Cerebellum

    Odd numbers = Left hemisphere (2001, 2101, etc.)
    Even numbers = Right hemisphere (2002, 2102, etc.)
    """

    # Frontal Lobe labels (2xxx)
    frontal_left = [
        2001,  # Precentral_L
        2101, 2102,  # Frontal_Sup_L, Frontal_Sup_Orb_L
        2201, 2202,  # Frontal_Mid_L, Frontal_Mid_Orb_L
        2301, 2302, 2311, 2312, 2321, 2322, 2331, 2332,  # Frontal_Inf variants
        2401, 2402,  # Rolandic_Oper_L, Supp_Motor_Area_L
        2501, 2502,  # Olfactory_L, Frontal_Sup_Medial_L
        2601, 2602, 2611, 2612,  # Frontal_Med_Orb_L, Rectus_L
        2701, 2702,  # Additional frontal
    ]

    frontal_right = [
        2002,  # Precentral_R
        2111, 2112,  # Frontal_Sup_R variants
        2211, 2212,  # Frontal_Mid_R variants
        # Mirror of left hemisphere odd→even
        2001 + 1, 2101 + 1, 2102 + 1, 2201 + 1, 2202 + 1,
        2301 + 1, 2302 + 1, 2311 + 1, 2312 + 1, 2321 + 1, 2322 + 1,
        2331 + 1, 2332 + 1, 2401 + 1, 2402 + 1, 2501 + 1, 2502 + 1,
        2601 + 1, 2602 + 1, 2611 + 1, 2612 + 1, 2701 + 1, 2702 + 1
    ]

    # Actually, let's use ranges for simplicity
    frontal_all = list(range(2000, 3000))  # All 2xxx labels are frontal

    # Parietal Lobe labels (6xxx)
    parietal_all = list(range(6000, 7000))  # All 6xxx labels are parietal

    # Temporal Lobe labels (8xxx)
    temporal_all = list(range(8000, 9000))  # All 8xxx labels are temporal

    # Occipital Lobe labels (5xxx)
    occipital_all = list(range(5000, 6000))  # All 5xxx labels are occipital

    # Insula (3xxx)
    insula_all = [3001, 3002]

    # Cingulate (3xxx)
    cingulate_all = [
        4001, 4002,  # Cingulum_Ant
        4011, 4012,  # Cingulum_Mid
        4021, 4022,  # Cingulum_Post
    ]

    # Basal Ganglia (7xxx)
    basal_ganglia_left = [7001, 7002, 7011, 7012, 7021, 7022]  # Caudate, Putamen, Pallidum
    basal_ganglia_right = [l + 1 for l in basal_ganglia_left if l % 2 == 1]
    basal_ganglia_all = list(range(7000, 7200))

    # Thalamus (7xxx)
    thalamus_all = [7101, 7102]

    # Hippocampus (4xxx)
    hippocampus_all = [4101, 4102]

    # Amygdala (4xxx)
    amygdala_all = [4201, 4202]

    # Brainstem - not specifically labeled in AAL, use thalamus as proxy
    brainstem_all = thalamus_all

    # Cerebellum (9xxx)
    cerebellum_all = list(range(9000, 10000))

    # Create comprehensive mapping
    region_mapping = {
        # Main cortical lobes
        "Frontal Lobe": {
            "Left": [l for l in frontal_all if l % 2 == 1 or l % 10 == 1],
            "Right": [l for l in frontal_all if l % 2 == 0 or l % 10 == 2],
            "Bilateral": frontal_all,
            "Unspecified": frontal_all
        },

        "Parietal Lobe": {
            "Left": [l for l in parietal_all if l % 2 == 1 or l % 10 == 1],
            "Right": [l for l in parietal_all if l % 2 == 0 or l % 10 == 2],
            "Bilateral": parietal_all,
            "Unspecified": parietal_all
        },

        "Temporal Lobe": {
            "Left": [l for l in temporal_all if l % 2 == 1 or l % 10 == 1],
            "Right": [l for l in temporal_all if l % 2 == 0 or l % 10 == 2],
            "Bilateral": temporal_all,
            "Unspecified": temporal_all
        },

        "Occipital Lobe": {
            "Left": [l for l in occipital_all if l % 2 == 1 or l % 10 == 1],
            "Right": [l for l in occipital_all if l % 2 == 0 or l % 10 == 2],
            "Bilateral": occipital_all,
            "Unspecified": occipital_all
        },

        # Subcortical structures
        "Basal Ganglia": {
            "Left": basal_ganglia_left,
            "Right": basal_ganglia_right,
            "Bilateral": basal_ganglia_all,
            "Unspecified": basal_ganglia_all
        },

        "Thalamus": {
            "Left": [7101],
            "Right": [7102],
            "Bilateral": thalamus_all,
            "Unspecified": thalamus_all
        },

        "Hippocampus": {
            "Left": [4101],
            "Right": [4102],
            "Bilateral": hippocampus_all,
            "Unspecified": hippocampus_all
        },

        "Amygdala": {
            "Left": [4201],
            "Right": [4202],
            "Bilateral": amygdala_all,
            "Unspecified": amygdala_all
        },

        "Insula": {
            "Left": [3001],
            "Right": [3002],
            "Bilateral": insula_all,
            "Unspecified": insula_all
        },

        "Brainstem": {
            "Left": thalamus_all,  # No specific brainstem in AAL
            "Right": thalamus_all,
            "Bilateral": thalamus_all,
            "Unspecified": thalamus_all
        },

        # Additional regions
        "Cingulate": {
            "Left": [4001, 4011, 4021],
            "Right": [4002, 4012, 4022],
            "Bilateral": cingulate_all,
            "Unspecified": cingulate_all
        },

        "Cerebellum": {
            "Left": [l for l in cerebellum_all if l % 2 == 1],
            "Right": [l for l in cerebellum_all if l % 2 == 0],
            "Bilateral": cerebellum_all,
            "Unspecified": cerebellum_all
        },

        # Ventricles - not in AAL, use whole brain
        "Lateral Ventricles": {
            "Left": frontal_all + parietal_all + temporal_all + occipital_all,
            "Right": frontal_all + parietal_all + temporal_all + occipital_all,
            "Bilateral": frontal_all + parietal_all + temporal_all + occipital_all,
            "Unspecified": frontal_all + parietal_all + temporal_all + occipital_all
        },

        # Whole brain fallback
        "Brain": {
            "Left": list(range(2000, 10000)),
            "Right": list(range(2000, 10000)),
            "Bilateral": list(range(2000, 10000)),
            "Unspecified": list(range(2000, 10000))
        },

        "Cerebral Hemisphere": {
            "Left": frontal_all + parietal_all + temporal_all + occipital_all,
            "Right": frontal_all + parietal_all + temporal_all + occipital_all,
            "Bilateral": frontal_all + parietal_all + temporal_all + occipital_all,
            "Unspecified": frontal_all + parietal_all + temporal_all + occipital_all
        },

        # Junction regions
        "Junction of Frontal and Parietal Lobes": {
            "Left": [l for l in (frontal_all + parietal_all) if l % 2 == 1],
            "Right": [l for l in (frontal_all + parietal_all) if l % 2 == 0],
            "Bilateral": frontal_all + parietal_all,
            "Unspecified": frontal_all + parietal_all
        },

        "Temporo-Parietal Region": {
            "Left": [l for l in (temporal_all + parietal_all) if l % 2 == 1],
            "Right": [l for l in (temporal_all + parietal_all) if l % 2 == 0],
            "Bilateral": temporal_all + parietal_all,
            "Unspecified": temporal_all + parietal_all
        },

        "Mesial Temporal Lobe": {
            "Left": [4101, 4102] + [l for l in temporal_all if l % 2 == 1],  # Hippocampus + temporal
            "Right": [4101, 4102] + [l for l in temporal_all if l % 2 == 0],
            "Bilateral": hippocampus_all + temporal_all,
            "Unspecified": hippocampus_all + temporal_all
        }
    }

    return region_mapping


def main():
    print("="*70)
    print("Creating AAL Region Mapping")
    print("="*70)

    # Create mapping
    region_mapping = create_aal_region_mapping()

    # Save
    output_path = "/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/region_mapping_aal.json"
    with open(output_path, 'w') as f:
        json.dump(region_mapping, f, indent=2)

    print(f"\n✓ Saved AAL region mapping to: {output_path}")

    # Print statistics
    print("\nRegion mapping statistics:")
    for region, sides in region_mapping.items():
        for side, labels in sides.items():
            num_labels = len(set(labels))  # Unique labels
            print(f"  {region:35s} ({side:11s}): {num_labels:3d} unique labels")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

    print("\nNext steps:")
    print("1. Update generate_sample_atlas_masks.py to use:")
    print("   --atlas_path /Disk1/afrouz/Data/TextBraTS_atlas_preprocess/brain_atlas_aal_resampled.nii.gz")
    print("   --region_mapping /Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/region_mapping_aal.json")
    print("2. Regenerate atlas masks with: python generate_sample_atlas_masks.py")
    print("3. Expected coverage: 15-40% (much better than 0.79%)")


if __name__ == "__main__":
    main()
