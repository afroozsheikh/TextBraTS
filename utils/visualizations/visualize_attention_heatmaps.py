"""
Visualize attention heatmaps before and after text-image fusion.
Similar to Figure 5 in the paper showing feature visualization overlaid on MRI images.

This script:
1. Loads a pretrained TextSwinUNETR model
2. Extracts intermediate features (enc3: before fusion, dec4: after fusion)
3. Upsamples features to original image size (128x128x128)
4. Overlays attention heatmaps on MRI slices
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.textswin_unetr import TextSwinUNETR
from monai import transforms
from monai.data import NibabelReader


class FeatureExtractorTextSwinUNETR(nn.Module):
    """
    Wrapper around TextSwinUNETR to extract intermediate features.

    Features extracted:
    - enc3: Before fusion features (shape: B, 192, 16, 16, 16)
    - dec4: After fusion features (shape: B, 768, 4, 4, 4)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.enc3_features = None
        self.dec4_features = None

    def forward(self, x_in, text_in, atlas_mask=None):
        """Forward pass with feature extraction."""
        # Get encoder features from SwinViT
        hidden_states_out = self.model.swinViT(x_in, text_in, self.model.normalize)

        # Encoder pathway
        enc0 = self.model.encoder1(x_in)
        enc1 = self.model.encoder2(hidden_states_out[0])
        enc2 = self.model.encoder3(hidden_states_out[1])

        # enc3: Before fusion features (1, 192, 16, 16, 16)
        enc3 = self.model.encoder4(hidden_states_out[2])
        self.enc3_features = enc3.detach()

        # dec4: After fusion features (1, 768, 4, 4, 4)
        dec4 = self.model.encoder10(hidden_states_out[4])
        self.dec4_features = dec4.detach()

        # Continue with decoder
        dec3 = self.model.decoder5(dec4, hidden_states_out[3])
        dec2 = self.model.decoder4(dec3, enc3)
        dec1 = self.model.decoder3(dec2, enc2)
        dec0 = self.model.decoder2(dec1, enc1)
        out = self.model.decoder1(dec0, enc0)
        logits = self.model.out(out)

        # Apply atlas mask if provided
        if atlas_mask is not None:
            alpha = torch.clamp(self.model.spatial_prompt_alpha, 0.0, 1.0)
            effective_mask = atlas_mask * alpha + (1.0 - alpha)
            logits = logits * effective_mask

        return logits, self.enc3_features, self.dec4_features


def reduce_channels_to_heatmap(features, method='mean'):
    """
    Reduce multi-channel features to single-channel heatmap.

    Args:
        features: Tensor of shape (B, C, H, W, D)
        method: 'mean', 'max', or 'attention'

    Returns:
        heatmap: Tensor of shape (B, 1, H, W, D)
    """
    if method == 'mean':
        # Average across channels
        heatmap = features.mean(dim=1, keepdim=True)
    elif method == 'max':
        # Max across channels
        heatmap = features.max(dim=1, keepdim=True)[0]
    elif method == 'attention':
        # Weighted attention: compute L2 norm across channels
        heatmap = torch.norm(features, p=2, dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    return heatmap


def upsample_to_size(features, target_size=(128, 128, 128)):
    """
    Upsample features to target size using trilinear interpolation.

    Args:
        features: Tensor of shape (B, C, H, W, D)
        target_size: Target spatial size (H, W, D)

    Returns:
        upsampled: Tensor of shape (B, C, target_H, target_W, target_D)
    """
    return F.interpolate(
        features,
        size=target_size,
        mode='trilinear',
        align_corners=True
    )


def normalize_heatmap(heatmap):
    """Normalize heatmap to [0, 1] range."""
    hmin = heatmap.min()
    hmax = heatmap.max()
    if hmax - hmin < 1e-8:
        return torch.zeros_like(heatmap)
    return (heatmap - hmin) / (hmax - hmin)


def load_model_and_weights(model_path, device='cuda'):
    """Load pretrained TextSwinUNETR model."""
    print(f"Loading model from {model_path}...")

    # Create model
    model = TextSwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        text_dim=768,
        use_checkpoint=False,
        use_text=True,
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model


def load_sample_data(data_dir, json_path, sample_idx=0):
    """
    Load a sample from the dataset.

    Returns:
        image: Tensor of shape (1, 4, 128, 128, 128)
        text_features: Tensor of shape (1, seq_len, 768)
        image_data: Original image data for visualization
        sample_info: Dict with sample metadata
    """
    print(f"Loading sample data...")

    # Load JSON
    with open(json_path, 'r') as f:
        data_list = json.load(f)

    


    # Get training samples
    samples = data_list['training']
    for entry in samples:
        entry["image"] = [img.replace(".gz", "") for img in entry.get("image", [])]
        if "label" in entry:
            entry["label"] = entry["label"].replace(".gz", "")



    sample = samples[sample_idx]

    print(f"Sample: {sample['image'][0]}")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], reader=NibabelReader()),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0,
            a_max=250.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=1,
        ),
    ])

    # Prepare data dict
    data_dict = {
        'image': [os.path.join(data_dir, img) for img in sample['image']],
        'label': os.path.join(data_dir, sample['label']),
    }

    # Apply transforms
    transformed = train_transform(data_dict)

    # RandCropByPosNegLabeld returns a list, so we need to extract the first element
    if isinstance(transformed, list):
        transformed = transformed[0]

    # Extract image and label
    image = transformed['image']  # Shape: (4, 128, 128, 128)
    label = transformed['label']  # Shape: (1, 128, 128, 128)

    # Load text features
    text_path = os.path.join(data_dir, sample['text_feature'])
    text_features = np.load(text_path)
    text_features = np.squeeze(text_features, axis=0)  # Remove extra dimension

    # Convert to tensors and add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # (1, 4, 128, 128, 128)
    text_tensor = torch.from_numpy(text_features).unsqueeze(0).float()  # (1, seq_len, 768)

    sample_info = {
        'sample_id': sample['image'][0].split('/')[0],
        'label': label,
    }

    print(f"✓ Data loaded: image {image_tensor.shape}, text {text_tensor.shape}")

    return image_tensor, text_tensor, image, sample_info


def create_custom_colormap():
    """Create a hot colormap for attention visualization."""
    colors = ['black', 'purple', 'red', 'orange', 'yellow', 'white']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
    return cmap


def visualize_attention_overlay(
    image_data,
    heatmap_before,
    heatmap_after,
    label_data,
    sample_id,
    output_path,
    slice_idx=None
):
    """
    Create Figure 5 style visualization with attention heatmaps overlaid on MRI.

    Args:
        image_data: MRI image (4, 128, 128, 128) - numpy array
        heatmap_before: Before fusion heatmap (1, 128, 128, 128) - numpy array
        heatmap_after: After fusion heatmap (1, 128, 128, 128) - numpy array
        label_data: Ground truth segmentation (1, 128, 128, 128) - numpy array
        sample_id: Sample identifier string
        output_path: Path to save the figure
        slice_idx: Specific slice index, or None for automatic selection
    """
    # Use T2 FLAIR (channel 0) for background
    background = image_data[0]  # (128, 128, 128)

    # Squeeze heatmaps
    heatmap_before = heatmap_before.squeeze()  # (128, 128, 128)
    heatmap_after = heatmap_after.squeeze()  # (128, 128, 128)
    label_data = label_data.squeeze()  # (128, 128, 128)

    # Find a good slice with tumor if not specified
    if slice_idx is None:
        tumor_slices = np.where(label_data.sum(axis=(0, 1)) > 0)[0]
        if len(tumor_slices) > 0:
            slice_idx = tumor_slices[len(tumor_slices) // 2]
        else:
            slice_idx = 64

    print(f"Visualizing slice {slice_idx}...")

    # Create custom colormap
    cmap_attention = create_custom_colormap()

    # Create figure
    fig = plt.figure(figsize=(18, 6))

    # Add main title
    fig.suptitle(
        f'Feature Visualization - {sample_id} (Axial Slice {slice_idx})',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    # Define subplot layout: 1 row, 3 columns
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3,
                          left=0.05, right=0.95, top=0.9, bottom=0.15)

    # Column 1: Original MRI with tumor segmentation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(background[:, :, slice_idx].T, cmap='gray', origin='lower')

    # Overlay tumor segmentation
    tumor_mask = label_data[:, :, slice_idx] > 0
    if tumor_mask.sum() > 0:
        tumor_overlay = np.ma.masked_where(~tumor_mask.T, tumor_mask.T)
        ax1.imshow(tumor_overlay, cmap='Reds', alpha=0.4, origin='lower')

    ax1.set_title('MRI (FLAIR) + Ground Truth', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Column 2: Before Fusion (enc3)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(background[:, :, slice_idx].T, cmap='gray', origin='lower', alpha=0.7)

    # Overlay before fusion heatmap
    heatmap_before_slice = heatmap_before[:, :, slice_idx].T
    im2 = ax2.imshow(heatmap_before_slice, cmap=cmap_attention, origin='lower',
                     alpha=0.6, vmin=0, vmax=1)
    ax2.set_title('Before Fusion\n(enc3: 192 channels, 16³)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Column 3: After Fusion (dec4)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(background[:, :, slice_idx].T, cmap='gray', origin='lower', alpha=0.7)

    # Overlay after fusion heatmap
    heatmap_after_slice = heatmap_after[:, :, slice_idx].T
    im3 = ax3.imshow(heatmap_after_slice, cmap=cmap_attention, origin='lower',
                     alpha=0.6, vmin=0, vmax=1)
    ax3.set_title('After Fusion\n(dec4: 768 channels, 4³)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Add colorbars
    cbar_ax2 = fig.add_axes([0.38, 0.08, 0.2, 0.03])
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Attention Intensity (Before)', fontsize=10)

    cbar_ax3 = fig.add_axes([0.68, 0.08, 0.2, 0.03])
    cbar3 = fig.colorbar(im3, cax=cbar_ax3, orientation='horizontal')
    cbar3.set_label('Attention Intensity (After)', fontsize=10)

    # Add legend for tumor
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.4, label='Tumor (Ground Truth)')
    ]
    fig.legend(handles=legend_elements, loc='lower left',
              bbox_to_anchor=(0.05, 0.02), fontsize=10)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    plt.close()


def visualize_multi_slice(
    image_data,
    heatmap_before,
    heatmap_after,
    label_data,
    sample_id,
    output_path,
    n_slices=5
):
    """
    Create multi-slice visualization showing multiple axial slices.
    """
    # Use T2 FLAIR (channel 0) for background
    background = image_data[0]  # (128, 128, 128)

    # Squeeze heatmaps
    heatmap_before = heatmap_before.squeeze()  # (128, 128, 128)
    heatmap_after = heatmap_after.squeeze()  # (128, 128, 128)
    label_data = label_data.squeeze()  # (128, 128, 128)

    # Find slices with tumor
    tumor_slices = np.where(label_data.sum(axis=(0, 1)) > 0)[0]
    if len(tumor_slices) > 0:
        # Select n_slices evenly spaced through tumor region
        indices = np.linspace(0, len(tumor_slices)-1, n_slices, dtype=int)
        slice_indices = tumor_slices[indices]
    else:
        # Fallback to center slices
        center = 64
        slice_indices = np.linspace(center-20, center+20, n_slices, dtype=int)

    # Create custom colormap
    cmap_attention = create_custom_colormap()

    # Create figure
    fig, axes = plt.subplots(n_slices, 3, figsize=(15, 4*n_slices))

    fig.suptitle(
        f'Multi-Slice Feature Visualization - {sample_id}',
        fontsize=16,
        fontweight='bold'
    )

    for i, slice_idx in enumerate(slice_indices):
        # Column 1: Original MRI with tumor
        axes[i, 0].imshow(background[:, :, slice_idx].T, cmap='gray', origin='lower')
        tumor_mask = label_data[:, :, slice_idx] > 0
        if tumor_mask.sum() > 0:
            tumor_overlay = np.ma.masked_where(~tumor_mask.T, tumor_mask.T)
            axes[i, 0].imshow(tumor_overlay, cmap='Reds', alpha=0.4, origin='lower')
        axes[i, 0].set_title(f'Slice {slice_idx}: MRI + GT', fontsize=11)
        axes[i, 0].axis('off')

        # Column 2: Before Fusion
        axes[i, 1].imshow(background[:, :, slice_idx].T, cmap='gray', origin='lower', alpha=0.7)
        axes[i, 1].imshow(heatmap_before[:, :, slice_idx].T, cmap=cmap_attention,
                         origin='lower', alpha=0.6, vmin=0, vmax=1)
        axes[i, 1].set_title(f'Before Fusion (enc3)', fontsize=11)
        axes[i, 1].axis('off')

        # Column 3: After Fusion
        axes[i, 2].imshow(background[:, :, slice_idx].T, cmap='gray', origin='lower', alpha=0.7)
        im = axes[i, 2].imshow(heatmap_after[:, :, slice_idx].T, cmap=cmap_attention,
                              origin='lower', alpha=0.6, vmin=0, vmax=1)
        axes[i, 2].set_title(f'After Fusion (dec4)', fontsize=11)
        axes[i, 2].axis('off')

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention Intensity', fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved multi-slice visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize attention heatmaps from TextSwinUNETR')
    parser.add_argument('--model_path', type=str,
                       default='/Disk1/afrouz/Projects/TextBraTS/runs/TextBraTS_conda/model.pt',
                       help='Path to pretrained model')
    parser.add_argument('--data_dir', type=str,
                       default='./data/TextBraTSData',
                       help='Path to dataset directory')
    parser.add_argument('--json_path', type=str,
                       default='./Train.json',
                       help='Path to dataset JSON file')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of sample to visualize')
    parser.add_argument('--output_dir', type=str,
                       default='./visualizations/attention_heatmaps',
                       help='Directory to save visualizations')
    parser.add_argument('--slice_idx', type=int, default=None,
                       help='Specific slice index to visualize (None for auto)')
    parser.add_argument('--reduction_method', type=str, default='attention',
                       choices=['mean', 'max', 'attention'],
                       help='Method to reduce channels to heatmap')
    parser.add_argument('--multi_slice', action='store_true',
                       help='Generate multi-slice visualization')
    parser.add_argument('--n_slices', type=int, default=5,
                       help='Number of slices for multi-slice visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("TextBraTS Attention Heatmap Visualization")
    print("="*80)

    # Load model
    model = load_model_and_weights(args.model_path, device=args.device)
    feature_extractor = FeatureExtractorTextSwinUNETR(model).to(args.device)

    # Load sample data
    image_tensor, text_tensor, image_data, sample_info = load_sample_data(
        args.data_dir, args.json_path, args.sample_idx
    )

    sample_id = sample_info['sample_id']
    label_data = sample_info['label']

    # Move to device
    image_tensor = image_tensor.to(args.device)
    text_tensor = text_tensor.to(args.device)

    print(f"\nExtracting features for {sample_id}...")

    # Forward pass to extract features
    with torch.no_grad():
        logits, enc3_features, dec4_features = feature_extractor(image_tensor, text_tensor)

    print(f"enc3 shape: {enc3_features.shape}")  # (1, 192, 16, 16, 16)
    print(f"dec4 shape: {dec4_features.shape}")  # (1, 768, 4, 4, 4)

    # Reduce channels to single heatmap
    print(f"\nReducing channels using method: {args.reduction_method}")
    heatmap_before = reduce_channels_to_heatmap(enc3_features, method=args.reduction_method)
    heatmap_after = reduce_channels_to_heatmap(dec4_features, method=args.reduction_method)

    # Upsample to original size (128, 128, 128)
    print("Upsampling features to 128³...")
    heatmap_before = upsample_to_size(heatmap_before, target_size=(128, 128, 128))
    heatmap_after = upsample_to_size(heatmap_after, target_size=(128, 128, 128))

    # Normalize heatmaps
    heatmap_before = normalize_heatmap(heatmap_before)
    heatmap_after = normalize_heatmap(heatmap_after)

    print(f"heatmap_before (upsampled) shape: {heatmap_before.shape}")
    print(f"heatmap_after (upsampled) shape: {heatmap_after.shape}")

    # Convert to numpy for visualization
    heatmap_before_np = heatmap_before.cpu().numpy()
    heatmap_after_np = heatmap_after.cpu().numpy()

    # Create visualizations
    if args.multi_slice:
        output_path = os.path.join(
            args.output_dir,
            f'{sample_id}_attention_multislice.png'
        )
        visualize_multi_slice(
            image_data,
            heatmap_before_np,
            heatmap_after_np,
            label_data,
            sample_id,
            output_path,
            n_slices=args.n_slices
        )
    else:
        output_path = os.path.join(
            args.output_dir,
            f'{sample_id}_attention_heatmap.png'
        )
        visualize_attention_overlay(
            image_data,
            heatmap_before_np,
            heatmap_after_np,
            label_data,
            sample_id,
            output_path,
            slice_idx=args.slice_idx
        )

    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()
