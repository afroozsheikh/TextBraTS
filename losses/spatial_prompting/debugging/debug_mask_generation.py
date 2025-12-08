"""
Debug script to identify why atlas masks are nearly empty
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path

# Paths
atlas_path = "/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/brain_atlas_harvard-oxford_resampled.nii.gz"
region_mapping_path = "/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/region_mapping_fixed.json"
volumetric_extractions_path = "/Disk1/afrouz/Projects/TextBraTS/losses/volumetric_extractions.json"

print("="*80)
print("ATLAS MASK GENERATION DEBUG")
print("="*80)

# Load atlas
print("\n1. Loading atlas...")
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
print(f"   Atlas shape: {atlas_data.shape}")

unique_labels = np.unique(atlas_data)
print(f"   Unique labels in atlas: {len(unique_labels)}")
print(f"   Label range: {unique_labels.min():.0f} to {unique_labels.max():.0f}")
print(f"   Labels: {unique_labels}")

# Load region mapping
print("\n2. Loading region mapping...")
with open(region_mapping_path, 'r') as f:
    region_mapping = json.load(f)

print(f"   Regions in mapping: {list(region_mapping.keys())}")

# Load volumetric extractions
print("\n3. Loading volumetric extractions...")
with open(volumetric_extractions_path, 'r') as f:
    volumetric_extractions = json.load(f)

# Test a specific sample
sample_id = "BraTS20_Training_001"
print(f"\n4. Testing sample: {sample_id}")
print("="*80)

sample_data = volumetric_extractions[sample_id]
pathologies = sample_data['Pathologies']

print("\n   Extracted pathologies:")
for path_type in ['Lesion', 'Edema', 'Necrosis']:
    entries = pathologies.get(path_type, [])
    print(f"\n   {path_type}: {len(entries)} entries")
    for i, entry in enumerate(entries):
        region = entry.get('Region', 'Unknown')
        side = entry.get('Side', 'Unspecified')
        print(f"      [{i+1}] {side} {region}")

# Test label retrieval
print("\n5. Testing label retrieval for Right Frontal Lobe...")
print("="*80)

# Check if "Frontal Lobe" is in region_mapping
if "Frontal Lobe" in region_mapping:
    print("   ✓ 'Frontal Lobe' found in region_mapping")

    if "Right" in region_mapping["Frontal Lobe"]:
        labels = region_mapping["Frontal Lobe"]["Right"]
        print(f"   ✓ 'Right' side found")
        print(f"   Labels for Right Frontal Lobe: {labels}")

        # Check if these labels exist in the atlas
        print(f"\n   Checking if these labels exist in atlas...")
        for label in labels:
            count = (atlas_data == label).sum()
            total = atlas_data.size
            percentage = count / total * 100
            exists = "✓" if count > 0 else "✗"
            print(f"      {exists} Label {label}: {count} voxels ({percentage:.4f}%)")

        # Create mask with these labels
        print(f"\n   Creating mask from these labels...")
        mask = np.zeros(atlas_data.shape, dtype=np.float32)
        for label in labels:
            mask[atlas_data == label] = 1.0

        mask_coverage = mask.sum() / mask.size * 100
        print(f"   Mask coverage: {mask_coverage:.4f}%")

        if mask_coverage == 0:
            print("   ⚠️  PROBLEM: Mask is completely empty!")
            print("   → The labels in region_mapping don't exist in the atlas")
    else:
        print("   ✗ 'Right' not found in Frontal Lobe mapping")
else:
    print("   ✗ 'Frontal Lobe' not found in region_mapping")

# Compare with actual atlas labels
print("\n6. Checking actual atlas label distribution...")
print("="*80)

print("\n   Voxel counts for all labels:")
for label in unique_labels:
    count = (atlas_data == label).sum()
    total = atlas_data.size
    percentage = count / total * 100
    if percentage > 0.01:  # Only show labels with >0.01% coverage
        print(f"      Label {int(label)}: {count:7d} voxels ({percentage:6.3f}%)")

# Check if the atlas is probabilistic or deterministic
print("\n7. Checking if atlas is probabilistic...")
print("="*80)

non_integer = np.any(atlas_data != atlas_data.astype(int))
if non_integer:
    print("   ⚠️  PROBLEM: Atlas contains non-integer values (probabilistic atlas)")
    print("   → The code expects integer labels, not probabilities")
    print(f"   → Value range: {atlas_data.min()} to {atlas_data.max()}")
    print(f"   → Sample values: {atlas_data.flatten()[:20]}")
else:
    print("   ✓ Atlas contains integer labels (deterministic)")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
