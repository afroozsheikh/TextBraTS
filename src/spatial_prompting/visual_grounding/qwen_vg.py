import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm

# GPU Selection Configuration
FORCE_CPU = False  # Set to True to force CPU usage
GPU_ID = 1  # Select which GPU to use (0, 1, 2, or 3)

# Check CUDA availability and setup device
def setup_device(gpu_id=0):
    """Setup the appropriate device for model inference"""
    if FORCE_CPU:
        print("Forcing CPU usage (FORCE_CPU=True)")
        return "cpu", torch.float32

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\nFound {num_gpus} GPU(s):")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # Validate GPU selection
        if gpu_id >= num_gpus:
            print(f"\nWarning: GPU {gpu_id} not available. Falling back to GPU 0")
            gpu_id = 0

        # Set the specific GPU
        torch.cuda.set_device(gpu_id)

        # Check CUDA compatibility on selected GPU
        try:
            # Test if CUDA operations work on this GPU
            test_tensor = torch.tensor([1.0]).cuda(gpu_id)
            del test_tensor
            torch.cuda.empty_cache()
            print(f"\nUsing GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return f"cuda:{gpu_id}", torch.float16
        except RuntimeError as e:
            print(f"CUDA error on GPU {gpu_id}: {e}")

            # Try other GPUs
            for other_gpu in range(num_gpus):
                if other_gpu == gpu_id:
                    continue
                try:
                    print(f"\nTrying GPU {other_gpu}...")
                    torch.cuda.set_device(other_gpu)
                    test_tensor = torch.tensor([1.0]).cuda(other_gpu)
                    del test_tensor
                    torch.cuda.empty_cache()
                    print(f"Success! Using GPU {other_gpu}: {torch.cuda.get_device_name(other_gpu)}")
                    return f"cuda:{other_gpu}", torch.float16
                except RuntimeError:
                    continue

            print("All GPUs failed. Falling back to CPU...")
            return "cpu", torch.float32
    else:
        print("CUDA not available. Using CPU...")
        return "cpu", torch.float32

device, dtype = setup_device(gpu_id=GPU_ID)

# 1. Load Model (Use Qwen2.5-VL-7B or Qwen3-VL)
# Note: "Qwen/Qwen2.5-VL-7B-Instruct" is recommended for 2D slices
print(f"\nLoading model on {device} with dtype {dtype}...")

# For specific GPU IDs (cuda:1, cuda:2, etc), we need to load without device_map
# and manually move the model to avoid device mismatch issues
if device.startswith("cuda:"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=dtype,
        device_map=None  # Don't use automatic device mapping for specific GPUs
    )
    model = model.to(device)  # Manually move to the specific GPU
elif device == "cpu":
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=dtype,
        device_map=None
    )
    model = model.to("cpu")
else:
    # device == "cuda" (default GPU)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=dtype,
        device_map="auto"
    )

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
print("Model loaded successfully!\n")

def normalize_slice(slice_data):
    """Convert raw MRI float data to a standard 0-255 RGB Image"""
    # Clip top 1% outliers to make the image brighter/clearer
    p99 = np.percentile(slice_data, 99)
    slice_data = np.clip(slice_data, 0, p99)
    
    # Normalize to 0-255
    if slice_data.max() > 0:
        slice_data = (slice_data / slice_data.max()) * 255
    
    img_uint8 = slice_data.astype(np.uint8)
    # Convert to RGB (3 channels)
    return Image.fromarray(np.stack([img_uint8]*3, axis=-1))

def find_lesion_in_3d(nifti_path, output_dir="visualizations"):
    # Load 3D Volume
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()

    # Create output directory for visualizations
    os.makedirs(output_dir, exist_ok=True)

    detections = []
    slice_images = []  # Store images for visualization

    # Iterate over slices (Axial plane usually index 2)
    # Step=2 to speed up (we don't need every single millimeter)
    for z_index in tqdm(range(0, volume.shape[2], 2), desc="Processing slices"):
        slice_data = volume[:, :, z_index]
        
        # Skip empty background slices
        if np.mean(slice_data) < 10: 
            continue
            
        image = normalize_slice(slice_data)
        
        # --- PREPARE QWEN INPUT ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Detect the brain tumor lesion."}
                ],
            }
        ]
        
        # Process Input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate Output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        print(f"Output text: {output_text}")
        
        # --- PARSE QWEN OUTPUT ---
        # Qwen output format: <|box_start|>(y1,x1),(y2,x2)<|box_end|>
        # Note: Qwen2.5 uses 0-1000 scale. We must convert back to pixels.
        if "<|box_start|>" in output_text:
            # Simple string parsing (or use regex)
            try:
                coords_str = output_text.split("<|box_start|>")[1].split("<|box_end|>")[0]
                # Coords look like "(y1,x1),(y2,x2)"
                c1, c2 = coords_str.split("),(")
                y1, x1 = map(int, c1.replace("(","").split(","))
                y2, x2 = map(int, c2.replace(")","").split(","))

                # Convert 1000-scale to Real Image Scale
                h, w = slice_data.shape
                real_x1 = int(x1 / 1000 * w)
                real_y1 = int(y1 / 1000 * h)
                real_x2 = int(x2 / 1000 * w)
                real_y2 = int(y2 / 1000 * h)

                detections.append([real_x1, real_y1, z_index, real_x2, real_y2, z_index+1])
                print(f"Slice {z_index}: Found lesion at ({real_x1},{real_y1}) to ({real_x2},{real_y2})")

                # Draw bounding box on image for visualization
                img_with_box = image.copy()
                draw = ImageDraw.Draw(img_with_box)
                draw.rectangle([real_x1, real_y1, real_x2, real_y2], outline="red", width=3)
                draw.text((real_x1, real_y1-10), f"Slice {z_index}", fill="red")

                # Save visualization
                img_with_box.save(os.path.join(output_dir, f"detection_slice_{z_index:03d}.png"))

                # Store for later visualization
                slice_images.append({
                    'z_index': z_index,
                    'image': img_with_box,
                    'bbox': (real_x1, real_y1, real_x2, real_y2),
                    'output_text': output_text
                })
            except Exception as e:
                print(f"Slice {z_index}: Parse error - {e}")
                pass # Parse error or empty box

    # --- AGGREGATE INTO 3D BOX ---
    if not detections:
        print("No lesion found.")
        return None, []

    detections = np.array(detections)
    # 3D Box = [min_x, min_y, min_z, max_x, max_y, max_z]
    bbox_3d = [
        int(np.min(detections[:, 0])), # x_min
        int(np.min(detections[:, 1])), # y_min
        int(np.min(detections[:, 2])), # z_min
        int(np.max(detections[:, 3])), # x_max
        int(np.max(detections[:, 4])), # y_max
        int(np.max(detections[:, 5]))  # z_max
    ]

    # Create summary visualization
    if slice_images:
        create_summary_visualization(slice_images, bbox_3d, output_dir)

    return bbox_3d, slice_images


def create_summary_visualization(slice_images, bbox_3d, output_dir):
    """Create a summary figure showing all detected slices"""
    num_slices = len(slice_images)
    if num_slices == 0:
        return

    # Create a grid layout
    cols = min(4, num_slices)
    rows = (num_slices + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle(f'Lesion Detections - 3D BBox: {bbox_3d}', fontsize=16)

    if num_slices == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, slice_info in enumerate(slice_images):
        axes[idx].imshow(slice_info['image'])
        axes[idx].set_title(f"Slice {slice_info['z_index']}\nBBox: {slice_info['bbox']}")
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_all_detections.png'), dpi=150, bbox_inches='tight')
    print(f"Summary visualization saved to {os.path.join(output_dir, 'summary_all_detections.png')}")
    plt.close()


def visualize_3d_bbox_on_volume(nifti_path, bbox_3d, output_dir="visualizations"):
    """Create 3D visualization showing the bounding box on the volume"""
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()

    if bbox_3d is None:
        return

    x_min, y_min, z_min, x_max, y_max, z_max = bbox_3d

    # Show the bounding box on three orthogonal views
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'3D Bounding Box Visualization\nBBox: {bbox_3d}', fontsize=16)

    # Axial view (z-slice at middle of bbox)
    z_mid = (z_min + z_max) // 2
    axial_slice = volume[:, :, z_mid]
    axes[0, 0].imshow(axial_slice.T, cmap='gray', origin='lower')
    axes[0, 0].add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                       fill=False, edgecolor='red', linewidth=2))
    axes[0, 0].set_title(f'Axial (z={z_mid})')
    axes[0, 0].axis('off')

    # Sagittal view (x-slice at middle of bbox)
    x_mid = (x_min + x_max) // 2
    sagittal_slice = volume[x_mid, :, :]
    axes[0, 1].imshow(sagittal_slice.T, cmap='gray', origin='lower')
    axes[0, 1].add_patch(plt.Rectangle((y_min, z_min), y_max-y_min, z_max-z_min,
                                       fill=False, edgecolor='red', linewidth=2))
    axes[0, 1].set_title(f'Sagittal (x={x_mid})')
    axes[0, 1].axis('off')

    # Coronal view (y-slice at middle of bbox)
    y_mid = (y_min + y_max) // 2
    coronal_slice = volume[:, y_mid, :]
    axes[0, 2].imshow(coronal_slice.T, cmap='gray', origin='lower')
    axes[0, 2].add_patch(plt.Rectangle((x_min, z_min), x_max-x_min, z_max-z_min,
                                       fill=False, edgecolor='red', linewidth=2))
    axes[0, 2].set_title(f'Coronal (y={y_mid})')
    axes[0, 2].axis('off')

    # Show max intensity projections
    axes[1, 0].imshow(np.max(volume, axis=2).T, cmap='gray', origin='lower')
    axes[1, 0].add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                       fill=False, edgecolor='red', linewidth=2))
    axes[1, 0].set_title('MIP Axial')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.max(volume, axis=0).T, cmap='gray', origin='lower')
    axes[1, 1].add_patch(plt.Rectangle((y_min, z_min), y_max-y_min, z_max-z_min,
                                       fill=False, edgecolor='red', linewidth=2))
    axes[1, 1].set_title('MIP Sagittal')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.max(volume, axis=1).T, cmap='gray', origin='lower')
    axes[1, 2].add_patch(plt.Rectangle((x_min, z_min), x_max-x_min, z_max-z_min,
                                       fill=False, edgecolor='red', linewidth=2))
    axes[1, 2].set_title('MIP Coronal')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_bbox_visualization.png'), dpi=150, bbox_inches='tight')
    print(f"3D bbox visualization saved to {os.path.join(output_dir, '3d_bbox_visualization.png')}")
    plt.close()

if __name__ == "__main__":
    # BraTS sample path
    sample_dir = "/Disk1/afrouz/Data/Merged/BraTS20_Training_002"

    # You can choose which modality to use: t1, t1ce, t2, or flair
    # t1ce (T1 contrast-enhanced) is often best for tumor visualization
    modality = "flair"  # Change to "t1", "t2", or "flair" as needed

    nifti_path = os.path.join(sample_dir, f"BraTS20_Training_002_{modality}.nii")
    output_dir = f"visualizations_{modality}"

    print(f"\n{'='*60}")
    print(f"Processing: {nifti_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Run lesion detection
    bbox_3d, slice_images = find_lesion_in_3d(nifti_path, output_dir=output_dir)

    # Print results
    print(f"\n{'='*60}")
    if bbox_3d:
        print("RESULTS:")
        print(f"Final 3D Bounding Box: {bbox_3d}")
        print(f"  - X range: {bbox_3d[0]} to {bbox_3d[3]}")
        print(f"  - Y range: {bbox_3d[1]} to {bbox_3d[4]}")
        print(f"  - Z range: {bbox_3d[2]} to {bbox_3d[5]}")
        print(f"  - Dimensions: {bbox_3d[3]-bbox_3d[0]} x {bbox_3d[4]-bbox_3d[1]} x {bbox_3d[5]-bbox_3d[2]}")
        print(f"  - Number of slices with detections: {len(slice_images)}")

        # Create 3D visualization
        print(f"\nCreating 3D visualization...")
        visualize_3d_bbox_on_volume(nifti_path, bbox_3d, output_dir=output_dir)

        print(f"\nAll visualizations saved to: {output_dir}/")
    else:
        print("No lesion detected.")
    print(f"{'='*60}\n")