"""
Analyze multiple samples to understand the tumor location vs text description mismatch.
"""

import json
import numpy as np
import nibabel as nib
from collections import Counter
from pathlib import Path
import random

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load volumetric extractions
with open("/Disk1/afrouz/Projects/TextBraTS/losses/volumetric_extractions.json") as f:
    extractions = json.load(f)

# Load region mapping
with open("/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/region_mapping_aal.json") as f:
    region_mapping = json.load(f)

# Load atlas
atlas = nib.load("/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/brain_atlas_aal_resampled.nii.gz").get_fdata()

# AAL label name mapping (simplified)
AAL_NAMES = {
    2000: "Frontal", 3000: "Insula/Cingulate", 4000: "Limbic",
    5000: "Occipital", 6000: "Parietal", 7000: "Basal Ganglia/Thalamus",
    8000: "Temporal", 9000: "Cerebellum"
}

def get_region_name(label):
    """Get region name from AAL label."""
    if label == 0:
        return "Background"
    for base, name in sorted(AAL_NAMES.items(), reverse=True):
        if label >= base:
            return name
    return "Unknown"

# Sample 20 random samples
sample_ids = random.sample(list(extractions.keys()), min(20, len(extractions)))

print("="*80)
print("TUMOR LOCATION vs TEXT DESCRIPTION ANALYSIS")
print("="*80)

results = []

for idx, sample_id in enumerate(sample_ids, 1):
    try:
        # Load ground truth
        gt_path = f"/Disk1/afrouz/Data/Merged/{sample_id}/{sample_id}_seg.nii"
        if not Path(gt_path).exists():
            continue

        gt = nib.load(gt_path).get_fdata()

        # Create tumor mask
        gt_wt = ((gt == 1) | (gt == 2) | (gt == 4)).astype(float)

        # Resize to 128x128x128
        from scipy.ndimage import zoom
        if gt.shape != (128, 128, 128):
            zoom_factors = [128/s for s in gt.shape]
            gt_wt = zoom(gt_wt, zoom_factors, order=0)

        if gt_wt.sum() == 0:
            continue

        # Find top atlas labels that overlap with tumor
        tumor_labels = []
        for label in np.unique(atlas):
            if label == 0:
                continue
            atlas_region = (atlas == int(label))
            overlap = (gt_wt * atlas_region).sum()
            if overlap > 0:
                pct = overlap / gt_wt.sum() * 100
                tumor_labels.append((int(label), overlap, pct))

        tumor_labels.sort(key=lambda x: x[1], reverse=True)

        # Get text-described regions
        text_regions = set()
        for pathology in ['Lesion', 'Edema']:
            entries = extractions[sample_id]['Pathologies'].get(pathology, [])
            for entry in entries:
                region = entry.get('Region', 'Unknown')
                text_regions.add(region)

        # Get labels that should be in mask based on text
        text_labels = set()
        for region in text_regions:
            if region in region_mapping:
                for side_labels in region_mapping[region].values():
                    text_labels.update(side_labels)

        # Compare
        top_5_labels = [l[0] for l in tumor_labels[:5]]
        top_5_in_text_mask = sum(1 for l in top_5_labels if l in text_labels)

        result = {
            'sample_id': sample_id,
            'text_regions': list(text_regions),
            'top_tumor_labels': tumor_labels[:5],
            'top_5_in_text_mask': top_5_in_text_mask,
            'total_labels_with_tumor': len(tumor_labels)
        }
        results.append(result)

        print(f"\n[{idx}/{len(sample_ids)}] {sample_id}")
        print(f"  Text regions: {', '.join(text_regions)}")
        print(f"  Top 5 actual tumor labels:")
        for label, overlap, pct in tumor_labels[:5]:
            region_name = get_region_name(label)
            in_mask = "✓" if label in text_labels else "✗"
            print(f"    {in_mask} Label {label:4d} ({region_name:20s}): {pct:5.1f}% of tumor")
        print(f"  Top 5 labels in text mask: {top_5_in_text_mask}/5")

    except Exception as e:
        print(f"\n[{idx}] Error processing {sample_id}: {e}")
        continue

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

if results:
    avg_top5_in_mask = np.mean([r['top_5_in_text_mask'] for r in results])
    print(f"\nAverage top-5 labels in text mask: {avg_top5_in_mask:.2f}/5")
    print(f"  → Only {avg_top5_in_mask/5*100:.1f}% of major tumor regions are in the text-based masks!")

    # Count how many labels typically overlap
    avg_labels_with_tumor = np.mean([r['total_labels_with_tumor'] for r in results])
    print(f"\nAverage atlas labels overlapping with tumor: {avg_labels_with_tumor:.1f}")

    # Most common actual tumor locations
    all_tumor_labels = []
    for r in results:
        all_tumor_labels.extend([l[0] for l in r['top_tumor_labels']])

    region_counter = Counter([get_region_name(l) for l in all_tumor_labels])
    print(f"\nMost common actual tumor regions (top 5):")
    for region, count in region_counter.most_common(5):
        print(f"  {region:30s}: {count:3d} occurrences")

    # Most common text-described regions
    all_text_regions = []
    for r in results:
        all_text_regions.extend(r['text_regions'])

    text_counter = Counter(all_text_regions)
    print(f"\nMost common text-described regions (top 5):")
    for region, count in text_counter.most_common(5):
        print(f"  {region:30s}: {count:3d} occurrences")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nThe text descriptions (e.g., 'Frontal Lobe', 'Parietal Lobe') are often")
print("too GENERAL and don't capture the actual tumor location which often includes:")
print("  - Deep structures (Basal Ganglia, Thalamus, Limbic)")
print("  - Adjacent regions not mentioned in text")
print("  - White matter tracts between named regions")
print("\nSOLUTION: When the text says 'Frontal Lobe', we need to include:")
print("  1. All frontal lobe subregions (not just a few)")
print("  2. Adjacent deep structures (Basal Ganglia, Thalamus)")
print("  3. White matter regions between cortical areas")
print("="*80)
