# BiomedParse Batch Inference for BraTS FLAIR Data

## Overview

This script performs batch inference using BiomedParse on BraTS FLAIR images with text prompts. It combines data loading, preprocessing (following the TextBraTS pipeline), and BiomedParse inference in a single workflow.

## Script: `biomedparse_batch_inference.py`

### What It Does

1. **Model Loading**
   - Loads BiomedParse model from HuggingFace (`microsoft/BiomedParse`)
   - Uses 3D configuration for volumetric medical images
   - Runs on GPU (CUDA device 2)

2. **Data Discovery**
   - Recursively finds all FLAIR files matching pattern `*_flair.nii`
   - Automatically loads corresponding text prompts from `*_flair_text.txt` files
   - Processes all samples in the data directory

3. **Preprocessing Pipeline** (matching TextBraTS main.py/data_utils.py)
   - Load images with MONAI NibabelReader
   - Resize to (128, 128, 128) ROI
   - Normalize intensity (nonzero=True, channel_wise=True)
   - Convert to tensors

4. **BiomedParse Inference**
   - Processes images through BiomedParse's `process_input` function
   - Runs model inference with text prompts
   - Post-processes predictions
   - Restores masks to original image dimensions

5. **Output Generation**
   - Saves predictions as NIfTI files (`.nii.gz`)
   - Creates visualization PNGs with 3 views (axial, coronal, sagittal)
   - Provides summary statistics

## Configuration

```python
DATA_DIR = "/Disk1/afrouz/Data/Merged"  # Input data directory
OUTPUT_DIR = "/Disk1/afrouz/Data/TextBraTS_biomedparse"  # Output directory
ROI_X = 128  # Same as TextBraTS main.py
ROI_Y = 128
ROI_Z = 128
BIOMEDPARSE_SIZE = 128  # BiomedParse processing size
SLICE_BATCH_SIZE = 4  # Batch size for slice processing
```

## Input Data Structure

Expected directory structure:
```
/Disk1/afrouz/Data/Merged/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii
│   └── BraTS20_Training_001_flair_text.txt
├── BraTS20_Training_002/
│   ├── BraTS20_Training_002_flair.nii
│   └── BraTS20_Training_002_flair_text.txt
└── ...
```

## Output Structure

```
/Disk1/afrouz/Data/TextBraTS_biomedparse/
├── BraTS20_Training_001_biomedparse_pred.nii.gz
├── BraTS20_Training_001_prediction.png
├── BraTS20_Training_002_biomedparse_pred.nii.gz
├── BraTS20_Training_002_prediction.png
└── ...
```

## Usage

### Basic Execution

```bash
conda activate biomedparse_v2
cd /Disk1/afrouz/Projects/TextBraTS/src/spatial_prompting/visual_grounding
python biomedparse_batch_inference.py
```

### Debugging in VSCode

1. Set up launch configuration (see below)
2. Open script in VSCode
3. Set breakpoints as needed
4. Press F5 or use "Run and Debug" panel

## Dependencies

- Python 3.10
- PyTorch with CUDA support
- BiomedParse (from `/Disk1/afrouz/Projects/BiomedParse`)
- MONAI
- nibabel
- hydra-core
- huggingface_hub
- matplotlib
- numpy

Install in conda environment:
```bash
conda activate biomedparse_v2
# Dependencies should already be installed in this environment
```

## Key Functions

### `find_all_flair_files(data_dir)`
Recursively searches for FLAIR files in the data directory.

### `preprocess_flair_for_main_pipeline(flair_path, roi_x, roi_y, roi_z)`
Applies MONAI preprocessing pipeline matching TextBraTS data_utils.py:
- LoadImage → Resize → NormalizeIntensity → ToTensor

### `run_biomedparse_inference(model, image, text_prompt, device, roi_size)`
Runs complete BiomedParse inference pipeline:
- Preprocessing for BiomedParse (128×128 slices)
- Model forward pass
- Post-processing and mask merging
- Restoration to original dimensions
- **Resizing to ROI size (128, 128, 128)**
- **Stacking 3 times to create (3, 128, 128, 128) output matching BraTS label format**

### `save_prediction(prediction, reference_nifti_path, output_path)`
Saves prediction as NIfTI with same affine/header as input.

### `visualize_prediction(image, prediction, sample_name, output_dir)`
Creates 2x3 grid visualization showing original and prediction overlay in 3 views.

## Processing Pipeline Details

### Step 1: Load FLAIR Image
- Uses nibabel to load `.nii` files
- Preserves original affine and header information

### Step 2: Load Text Prompt
- Reads corresponding `*_flair_text.txt` file
- Text describes lesion location and characteristics

### Step 3: BiomedParse Processing
- Converts image to 128x128 slices (updated from 512)
- Applies padding and normalization
- Processes through 3D BiomedParse model
- Uses text prompt to guide segmentation

### Step 4: Post-processing
- Applies threshold and NMS
- Merges multi-class predictions
- Restores to original image dimensions
- **Resizes to (128, 128, 128) using MONAI Resize transform**
- **Stacks mask 3 times to create (3, 128, 128, 128) shape matching BraTS labels**

### Step 5: Save Results
- NIfTI file with prediction mask (shape: 3×128×128×128)
- PNG visualization with overlay

## Output Statistics

For each sample, the script reports:
- Original image shape and data range
- Preprocessed shape
- Prediction shape
- Number of positive voxels (predicted tumor)
- Unique values in prediction

## Error Handling

- Continues processing if individual samples fail
- Logs errors for each failed sample
- Provides summary of successful vs. failed samples
- Skips samples without text prompts

## Performance Notes

- GPU memory usage depends on image size and batch size
- Adjust `SLICE_BATCH_SIZE` if running out of memory
- Processing time: ~10-30 seconds per sample (depending on GPU)

## Comparison to TextBraTS Pipeline

| Component | TextBraTS (main.py) | BiomedParse Script |
|-----------|---------------------|-------------------|
| Data Loading | MONAI Dataset with JSON | Direct file search |
| Preprocessing | MONAI transforms | Same MONAI transforms |
| ROI Size | (128, 128, 128) | (128, 128, 128) |
| Normalization | Nonzero, channel-wise | Same |
| Model | TextSwinUNETR | BiomedParse |
| Text Integration | BioBERT embeddings | Direct text prompts |

## Troubleshooting

### Issue: No FLAIR files found
- Check `DATA_DIR` path is correct
- Ensure files match pattern `*_flair.nii`
- Verify directory structure

### Issue: CUDA out of memory
- Reduce `SLICE_BATCH_SIZE` (try 2 or 1)
- Process fewer samples at once
- Use smaller `BIOMEDPARSE_SIZE`

### Issue: Missing text prompts
- Ensure `*_flair_text.txt` files exist
- Script will skip samples without text prompts

### Issue: Model loading fails
- Check BiomedParse path: `/Disk1/afrouz/Projects/BiomedParse`
- Verify HuggingFace cache has model weights
- Check internet connection for first-time download

## Future Enhancements

- [ ] Add support for multiple modalities (T1, T1ce, T2)
- [ ] Batch processing of multiple samples in parallel
- [ ] Quantitative evaluation against ground truth labels
- [ ] Integration with TextBraTS training pipeline
- [ ] Support for different atlas mask integration
- [ ] Multi-GPU support

## References

- BiomedParse: https://github.com/microsoft/BiomedParse
- MONAI: https://monai.io/
- TextBraTS: Custom implementation for text-guided BraTS segmentation

## Author

Created for TextBraTS project - BiomedParse integration
Date: 2026-01-04
