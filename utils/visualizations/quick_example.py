"""
Quick example script demonstrating how to use the visualization tools.
This is a simplified version for quick testing.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from utils.visualizations.visualize_attention_heatmaps import (
    load_model_and_weights,
    load_sample_data,
    FeatureExtractorTextSwinUNETR,
    reduce_channels_to_heatmap,
    upsample_to_size,
    normalize_heatmap,
    visualize_attention_overlay,
    visualize_multi_slice
)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def quick_visualize(
    model_path='/Disk1/afrouz/Projects/TextBraTS/runs/TextBraTS_conda/model.pt',
    data_dir='/Disk1/afrouz/Data/Merged/',
    json_path='./Train.json',
    sample_idx=0,
    output_dir='.',
    device='cuda'
):
    """
    Quick visualization example showing the entire pipeline.
    """
    print("="*80)
    print("Quick Attention Heatmap Visualization Example")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load model
    print("\n[1/5] Loading pretrained model...")
    model = load_model_and_weights(model_path, device=device)
    feature_extractor = FeatureExtractorTextSwinUNETR(model).to(device)

    # Step 2: Load sample data
    print("\n[2/5] Loading sample data...")
    image_tensor, text_tensor, image_data, sample_info = load_sample_data(
        data_dir, json_path, sample_idx
    )
    sample_id = sample_info['sample_id']
    label_data = sample_info['label']

    # Move to device
    image_tensor = image_tensor.to(device)
    text_tensor = text_tensor.to(device)

    # Step 3: Extract features
    print(f"\n[3/5] Extracting features from {sample_id}...")
    with torch.no_grad():
        logits, enc3_features, dec4_features = feature_extractor(image_tensor, text_tensor)

    print(f"  - enc3 (before fusion): {enc3_features.shape}")
    print(f"  - dec4 (after fusion):  {dec4_features.shape}")

    # Step 4: Process features into heatmaps
    print("\n[4/5] Processing features into heatmaps...")

    # Reduce channels
    heatmap_before = reduce_channels_to_heatmap(enc3_features, method='attention')
    heatmap_after = reduce_channels_to_heatmap(dec4_features, method='attention')

    # Upsample to 128x128x128
    heatmap_before = upsample_to_size(heatmap_before, target_size=(128, 128, 128))
    heatmap_after = upsample_to_size(heatmap_after, target_size=(128, 128, 128))

    # Normalize
    heatmap_before = normalize_heatmap(heatmap_before)
    heatmap_after = normalize_heatmap(heatmap_after)

    # Convert to numpy
    heatmap_before_np = heatmap_before.cpu().numpy()
    heatmap_after_np = heatmap_after.cpu().numpy()

    print(f"  - Heatmaps upsampled to: {heatmap_before_np.shape}")

    # Step 5: Create visualizations
    print("\n[5/5] Creating visualizations...")

    # Single slice visualization
    output_path_single = os.path.join(output_dir, f'{sample_id}_single_slice.png')
    visualize_attention_overlay(
        image_data,
        heatmap_before_np,
        heatmap_after_np,
        label_data,
        sample_id,
        output_path_single,
        slice_idx=None  # Auto-select tumor slice
    )

    # Multi-slice visualization
    output_path_multi = os.path.join(output_dir, f'{sample_id}_multi_slice.png')
    visualize_multi_slice(
        image_data,
        heatmap_before_np,
        heatmap_after_np,
        label_data,
        sample_id,
        output_path_multi,
        n_slices=5
    )

    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  - Single slice: {output_path_single}")
    print(f"  - Multi-slice:  {output_path_multi}")
    print("\nYou can now view these images to see the attention heatmaps")
    print("overlaid on the MRI scans, similar to Figure 5 in the paper.")
    print("="*80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Quick visualization example')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize (default: 0)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    quick_visualize(sample_idx=args.sample_idx, device=args.device)
