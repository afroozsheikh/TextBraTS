"""
Complete script for BiomedParse inference on BraTS FLAIR data
- Loads BiomedParse model
- Reads all FLAIR files from data directory
- Performs preprocessing (similar to main.py/data_utils.py)
- Runs BiomedParse inference with text prompts
- Saves predictions
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
sys.path.insert(0, '/Disk1/afrouz/Projects/BiomedParse')

import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
from typing import List, Dict, Tuple

# BiomedParse imports
from utils import process_input, process_output, slice_nms
from inference import postprocess, merge_multiclass_masks

# MONAI imports for preprocessing (same as data_utils.py)
from monai import transforms
from monai.data import NibabelReader


# ==================== Configuration ====================
DATA_DIR = "/Disk1/afrouz/Data/Merged"  # Root directory containing BraTS data
OUTPUT_DIR = "/Disk1/afrouz/Data/TextBraTS_biomedparse"
ROI_X = 128  # Same as main.py
ROI_Y = 128
ROI_Z = 128
BIOMEDPARSE_SIZE = 128  # BiomedParse expects 128x128 slices
SLICE_BATCH_SIZE = 4  # Number of slices to process at once


# ==================== Helper Functions ====================

def find_all_flair_files(data_dir: str, pattern: str = "*_flair.nii") -> List[str]:
    """Find all FLAIR files in the data directory"""
    search_pattern = os.path.join(data_dir, "**", pattern)
    flair_files = glob.glob(search_pattern, recursive=True)
    flair_files.sort()
    print(f"Found {len(flair_files)} FLAIR files in {data_dir}")
    return flair_files


def load_text_prompt(flair_path: str) -> str:
    """Load corresponding text prompt for a FLAIR image"""
    text_path = flair_path.replace("_flair.nii", "_flair_text.txt")

    if os.path.exists(text_path):
        with open(text_path, 'r') as f:
            return f.read().strip()
    else:
        print(f"Warning: Text file not found at {text_path}")
        return ""


def preprocess_flair_for_main_pipeline(flair_path: str,
                                       roi_x: int = 128,
                                       roi_y: int = 128,
                                       roi_z: int = 128) -> torch.Tensor:
    """
    Preprocess FLAIR image following the same pipeline as main.py/data_utils.py

    Pipeline:
    1. LoadImage with NibabelReader
    2. Resize to (roi_x, roi_y, roi_z)
    3. NormalizeIntensity (nonzero, channel_wise)
    4. ToTensor
    """
    transform = transforms.Compose([
        transforms.LoadImage(reader=NibabelReader(), image_only=True),
        transforms.Resize(spatial_size=[roi_x, roi_y, roi_z]),
        transforms.NormalizeIntensity(nonzero=True, channel_wise=True),
        transforms.ToTensor(),
    ])

    return transform(flair_path)


def preprocess_for_biomedparse(image: np.ndarray, target_size: int = 512) -> Tuple:
    """
    Preprocess image for BiomedParse model using their process_input function

    Args:
        image: numpy array of shape (H, W, D)
        target_size: target size for BiomedParse (512)

    Returns:
        imgs: preprocessed tensor
        pad_width: padding information
        padded_size: size after padding
        valid_axis: valid axis
    """
    imgs, pad_width, padded_size, valid_axis = process_input(image, target_size)
    return imgs, pad_width, padded_size, valid_axis


def run_biomedparse_inference(model,
                              image: np.ndarray,
                              text_prompt: str,
                              device,
                              target_size: int = 512,
                              slice_batch_size: int = 4,
                              roi_size: tuple = (128, 128, 128)) -> np.ndarray:
    """
    Run BiomedParse inference on a single FLAIR image

    Args:
        model: BiomedParse model
        image: numpy array (H, W, D)
        text_prompt: text description
        device: torch device
        target_size: target size for BiomedParse
        slice_batch_size: batch size for slice processing
        roi_size: target ROI size (default: (128, 128, 128))

    Returns:
        final_mask: predicted mask of shape (3, 128, 128, 128) matching BraTS label format
    """
    # Preprocess for BiomedParse
    imgs, pad_width, padded_size, valid_axis = preprocess_for_biomedparse(image, target_size)
    imgs = imgs.to(device).int()

    # Prepare input
    input_tensor = {
        "image": imgs.unsqueeze(0),
        "text": [text_prompt],
    }

    print(f"  BiomedParse input shape: {imgs.shape}")

    # Run inference
    with torch.no_grad():
        output = model(input_tensor, mode="eval", slice_batch_size=slice_batch_size)

    # Get mask predictions
    mask_preds = output["predictions"]["pred_gmasks"]
    print(f"  Raw predictions shape: {mask_preds.shape}")

    # Interpolate to target size
    mask_preds = F.interpolate(
        mask_preds,
        size=(target_size, target_size),
        mode="bicubic",
        align_corners=False,
        antialias=True
    )

    # Postprocess
    mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])

    # Merge multi-class masks (BraTS has single tumor class, use id 1)
    ids = [1]
    mask_preds = merge_multiclass_masks(mask_preds, ids)

    # Process output to original dimensions
    final_mask = process_output(mask_preds, pad_width, padded_size, valid_axis)
    print(f"  Processed mask shape: {final_mask.shape}")

    # Resize to ROI size (128, 128, 128) using MONAI transform
    # Convert to tensor for MONAI resize
    final_mask_tensor = torch.from_numpy(final_mask).unsqueeze(0).float()  # Add channel dim

    # Resize to (128, 128, 128)
    resize_transform = transforms.Resize(spatial_size=roi_size)
    resized_mask = resize_transform(final_mask_tensor)

    # Convert back to numpy and remove channel dim
    resized_mask = resized_mask.squeeze(0).numpy()  # Shape: (128, 128, 128)

    # Stack 3 times to match BraTS label format (3, 128, 128, 128)
    # This creates 3 channels for TC, WT, ET (all same for now)
    final_mask_3ch = np.stack([resized_mask, resized_mask, resized_mask], axis=0)

    print(f"  Final mask shape (3 channels): {final_mask_3ch.shape}")

    return final_mask_3ch


def save_prediction(prediction: np.ndarray,
                   reference_nifti_path: str,
                   output_path: str):
    """Save prediction as NIfTI file with same affine as reference"""
    ref_nii = nib.load(reference_nifti_path)
    pred_nii = nib.Nifti1Image(prediction.astype(np.float32), ref_nii.affine, ref_nii.header)
    nib.save(pred_nii, output_path)
    print(f"  Saved prediction to {output_path}")


def visualize_prediction(image: np.ndarray,
                        prediction: np.ndarray,
                        sample_name: str,
                        output_dir: str):
    """
    Create visualization of prediction overlaid on image

    Args:
        image: Original image (H, W, D) - will be resized to match prediction
        prediction: Prediction mask (3, 128, 128, 128) or (128, 128, 128)
        sample_name: Sample name for title
        output_dir: Output directory for visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Handle 3-channel prediction by taking first channel for visualization
    if prediction.ndim == 4 and prediction.shape[0] == 3:
        pred_vis = prediction[0]  # Use first channel (WT) for visualization
    else:
        pred_vis = prediction

    # Resize image to match prediction size for visualization
    # Convert image to tensor and resize
    image_tensor = torch.from_numpy(image).unsqueeze(0).float()
    resize_transform = transforms.Resize(spatial_size=[128, 128, 128])
    image_resized = resize_transform(image_tensor).squeeze(0).numpy()

    # Get middle slices
    slice_idx_axial = pred_vis.shape[2] // 2
    slice_idx_coronal = pred_vis.shape[1] // 2
    slice_idx_sagittal = pred_vis.shape[0] // 2

    # Original images (resized)
    axes[0, 0].imshow(image_resized[:, :, slice_idx_axial], cmap='gray')
    axes[0, 0].set_title(f'FLAIR - Axial (slice {slice_idx_axial})')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image_resized[:, slice_idx_coronal, :], cmap='gray')
    axes[0, 1].set_title(f'FLAIR - Coronal (slice {slice_idx_coronal})')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(image_resized[slice_idx_sagittal, :, :], cmap='gray')
    axes[0, 2].set_title(f'FLAIR - Sagittal (slice {slice_idx_sagittal})')
    axes[0, 2].axis('off')

    # Predictions overlaid
    axes[1, 0].imshow(image_resized[:, :, slice_idx_axial], cmap='gray')
    axes[1, 0].imshow(pred_vis[:, :, slice_idx_axial], cmap='jet', alpha=0.5)
    axes[1, 0].set_title('Prediction - Axial')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(image_resized[:, slice_idx_coronal, :], cmap='gray')
    axes[1, 1].imshow(pred_vis[:, slice_idx_coronal, :], cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Prediction - Coronal')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(image_resized[slice_idx_sagittal, :, :], cmap='gray')
    axes[1, 2].imshow(pred_vis[slice_idx_sagittal, :, :], cmap='jet', alpha=0.5)
    axes[1, 2].set_title('Prediction - Sagittal')
    axes[1, 2].axis('off')

    plt.suptitle(f'{sample_name} - BiomedParse Prediction', fontsize=16)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, f"{sample_name}_prediction.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {fig_path}")


# ==================== Main Processing ====================

def main():
    print("="*80)
    print("BiomedParse Batch Inference for BraTS FLAIR Data")
    print("="*80)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Preprocessing ROI: ({ROI_X}, {ROI_Y}, {ROI_Z})")
    print(f"BiomedParse size: {BIOMEDPARSE_SIZE}")
    print("="*80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize BiomedParse model
    print("\n[1/4] Loading BiomedParse model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../../../../BiomedParse/configs/model",
                    job_name="batch_prediction",
                    version_base=None)
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")

    from huggingface_hub import hf_hub_download
    model.load_pretrained(hf_hub_download(
        repo_id="microsoft/BiomedParse", filename="biomedparse_v2.ckpt"))
    model = model.to(device).eval()
    print("Model loaded successfully!")

    # Find all FLAIR files
    print("\n[2/4] Finding FLAIR files...")
    flair_files = find_all_flair_files(DATA_DIR)

    if len(flair_files) == 0:
        print(f"ERROR: No FLAIR files found in {DATA_DIR}")
        return

    # Process each file
    print(f"\n[3/4] Processing {len(flair_files)} FLAIR files...")
    print("="*80)

    results = []

    for i, flair_path in enumerate(flair_files):
        sample_name = Path(flair_path).parent.name
        print(f"\n[{i+1}/{len(flair_files)}] Processing {sample_name}...")
        print(f"  Path: {flair_path}")

        try:
            # Load raw FLAIR image
            flair_img = nib.load(flair_path)
            image = flair_img.get_fdata()
            print(f"  Raw shape: {image.shape}, dtype: {image.dtype}")
            print(f"  Raw range: [{image.min():.2f}, {image.max():.2f}]")

            # Load text prompt
            text_prompt = load_text_prompt(flair_path)
            if not text_prompt:
                print(f"  WARNING: No text prompt found, skipping...")
                continue
            print(f"  Text prompt: {text_prompt[:100]}...")

            # Run BiomedParse inference
            prediction = run_biomedparse_inference(
                model=model,
                image=image,
                text_prompt=text_prompt,
                device=device,
                target_size=BIOMEDPARSE_SIZE,
                slice_batch_size=SLICE_BATCH_SIZE,
                roi_size=(ROI_X, ROI_Y, ROI_Z)
            )

            # Print prediction statistics
            unique_vals = np.unique(prediction)
            print(f"  Unique values in prediction: {unique_vals}")
            print(f"  Positive voxels: {np.sum(prediction > 0)} / {prediction.size}")

            # Save prediction as NIfTI
            pred_path = os.path.join(OUTPUT_DIR, f"{sample_name}_biomedparse_pred.nii.gz")
            save_prediction(prediction, flair_path, pred_path)

            # Create visualization
            if i < 10:
                visualize_prediction(image, prediction, sample_name, OUTPUT_DIR)

            results.append({
                'sample_name': sample_name,
                'flair_path': flair_path,
                'prediction_path': pred_path,
                'original_shape': image.shape,
                'prediction_shape': prediction.shape,
                'positive_voxels': np.sum(prediction > 0),
                'status': 'success'
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'sample_name': sample_name,
                'flair_path': flair_path,
                'status': 'failed',
                'error': str(e)
            })
            continue

    # Summary
    print("\n" + "="*80)
    print("[4/4] Summary")
    print("="*80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"Total files: {len(flair_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSuccessfully processed samples:")
        for r in successful[:10]:  # Show first 10
            print(f"  - {r['sample_name']}: {r['positive_voxels']} positive voxels")
        if len(successful) > 10:
            print(f"  ... and {len(successful) - 10} more")

    if failed:
        print("\nFailed samples:")
        for r in failed:
            print(f"  - {r['sample_name']}: {r.get('error', 'Unknown error')}")

    print(f"\nAll predictions saved to: {OUTPUT_DIR}")
    print("="*80)

    return results


if __name__ == "__main__":
    results = main()
