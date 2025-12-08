"""
Visualize where the tumor is vs where the atlas mask thinks it should be.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SAMPLE_ID = "BraTS20_Training_001"

# Load ground truth
gt = nib.load(f"/Disk1/afrouz/Data/Merged/{SAMPLE_ID}/{SAMPLE_ID}_seg.nii").get_fdata()

# Create tumor masks
gt_wt = ((gt == 1) | (gt == 2) | (gt == 4)).astype(float)
gt_tc = ((gt == 1) | (gt == 4)).astype(float)
gt_et = (gt == 4).astype(float)

# Resize to 128x128x128
from scipy.ndimage import zoom
zoom_factors = [128/s for s in gt.shape]
gt_wt = zoom(gt_wt, zoom_factors, order=0)
gt_tc = zoom(gt_tc, zoom_factors, order=0)
gt_et = zoom(gt_et, zoom_factors, order=0)

# Load atlas mask
mask = nib.load(f"/Disk1/afrouz/Data/TextBraTS_atlas_masks_aal/{SAMPLE_ID}_atlas_mask.nii.gz").get_fdata()

# Load actual atlas to see all labels
atlas = nib.load("/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/brain_atlas_aal_resampled.nii.gz").get_fdata()

# Find tumor center of mass
tumor_coords = np.where(gt_wt > 0)
if len(tumor_coords[0]) > 0:
    tumor_center_x = int(np.mean(tumor_coords[0]))
    tumor_center_y = int(np.mean(tumor_coords[1]))
    tumor_center_z = int(np.mean(tumor_coords[2]))
    print(f"Tumor center of mass: ({tumor_center_x}, {tumor_center_y}, {tumor_center_z})")

    # What atlas label is at the tumor center?
    atlas_label_at_center = int(atlas[tumor_center_x, tumor_center_y, tumor_center_z])
    print(f"Atlas label at tumor center: {atlas_label_at_center}")

    # Check if this label is in the mask
    if atlas_label_at_center > 0:
        # Check which channel masks contain this label
        atlas_mask_at_center = atlas == atlas_label_at_center
        tc_has_it = np.any(mask[0, atlas_mask_at_center] > 0)
        wt_has_it = np.any(mask[1, atlas_mask_at_center] > 0)
        et_has_it = np.any(mask[2, atlas_mask_at_center] > 0)
        print(f"Is this label in atlas masks? TC={tc_has_it}, WT={wt_has_it}, ET={et_has_it}")
    else:
        print(f"⚠️  Tumor center is in background (label 0)!")

# Find all atlas labels that overlap with tumor
tumor_labels = []
for label in range(1, 10000):
    atlas_region = (atlas == label)
    if atlas_region.sum() == 0:
        continue
    overlap = (gt_wt * atlas_region).sum()
    if overlap > 0:
        tumor_labels.append((label, overlap))

tumor_labels.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop 10 atlas labels that overlap with tumor:")
for label, overlap in tumor_labels[:10]:
    pct = overlap / gt_wt.sum() * 100
    print(f"  Label {label}: {int(overlap):,} voxels ({pct:.1f}% of tumor)")

# Create visualization
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle(f'{SAMPLE_ID} - Tumor vs Atlas Mask Mismatch Analysis', fontsize=14, fontweight='bold')

# Find a slice with tumor
slice_z = tumor_center_z if len(tumor_coords[0]) > 0 else 64

# Row 1: Axial view
axes[0, 0].imshow(gt_wt[:, :, slice_z].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
axes[0, 0].set_title('GT: Whole Tumor')
axes[0, 0].axis('off')

axes[0, 1].imshow(mask[1, :, :, slice_z].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
axes[0, 1].set_title('Atlas Mask: WT')
axes[0, 1].axis('off')

# Overlay
axes[0, 2].imshow(gt_wt[:, :, slice_z].T, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
axes[0, 2].imshow(mask[1, :, :, slice_z].T, cmap='Greens', origin='lower', alpha=0.5, vmin=0, vmax=1)
axes[0, 2].set_title('Overlay: Red=GT, Green=Atlas')
axes[0, 2].axis('off')

# Atlas labels
axes[0, 3].imshow(atlas[:, :, slice_z].T, cmap='tab20', origin='lower')
axes[0, 3].set_title('Atlas Labels')
axes[0, 3].axis('off')

# Row 2: Coronal view
slice_y = tumor_center_y if len(tumor_coords[0]) > 0 else 64

axes[1, 0].imshow(gt_wt[:, slice_y, :].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
axes[1, 0].set_title('GT: Whole Tumor (Coronal)')
axes[1, 0].axis('off')

axes[1, 1].imshow(mask[1, :, slice_y, :].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
axes[1, 1].set_title('Atlas Mask: WT (Coronal)')
axes[1, 1].axis('off')

axes[1, 2].imshow(gt_wt[:, slice_y, :].T, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
axes[1, 2].imshow(mask[1, :, slice_y, :].T, cmap='Greens', origin='lower', alpha=0.5, vmin=0, vmax=1)
axes[1, 2].set_title('Overlay (Coronal)')
axes[1, 2].axis('off')

axes[1, 3].imshow(atlas[:, slice_y, :].T, cmap='tab20', origin='lower')
axes[1, 3].set_title('Atlas Labels (Coronal)')
axes[1, 3].axis('off')

# Row 3: Sagittal view
slice_x = tumor_center_x if len(tumor_coords[0]) > 0 else 64

axes[2, 0].imshow(gt_wt[slice_x, :, :].T, cmap='Reds', origin='lower', vmin=0, vmax=1)
axes[2, 0].set_title('GT: Whole Tumor (Sagittal)')
axes[2, 0].axis('off')

axes[2, 1].imshow(mask[1, slice_x, :, :].T, cmap='Greens', origin='lower', vmin=0, vmax=1)
axes[2, 1].set_title('Atlas Mask: WT (Sagittal)')
axes[2, 1].axis('off')

axes[2, 2].imshow(gt_wt[slice_x, :, :].T, cmap='Reds', origin='lower', alpha=0.5, vmin=0, vmax=1)
axes[2, 2].imshow(mask[1, slice_x, :, :].T, cmap='Greens', origin='lower', alpha=0.5, vmin=0, vmax=1)
axes[2, 2].set_title('Overlay (Sagittal)')
axes[2, 2].axis('off')

axes[2, 3].imshow(atlas[slice_x, :, :].T, cmap='tab20', origin='lower')
axes[2, 3].set_title('Atlas Labels (Sagittal)')
axes[2, 3].axis('off')

plt.tight_layout()
output_path = '/Disk1/afrouz/Projects/TextBraTS/losses/spatial_prompting/debugging/tumor_atlas_mismatch.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to {output_path}")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
