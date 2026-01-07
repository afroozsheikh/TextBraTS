# Enriched Text Features for TextBraTS

## Overview

This document explains the enriched text feature implementation and the rationale behind augmenting the original text descriptions with quantitative volumetric and count information.

## Background

### Initial Assumption
We initially assumed that the TextBraTS model was already handling spatial location information through the text descriptions. The original FLAIR text descriptions provided qualitative imaging characteristics such as:
- Signal intensity patterns
- Enhancement characteristics
- Border characteristics
- Tumor location and appearance

### The Problem
However, the original text descriptions lacked **quantitative measurements** that are critical for precise tumor segmentation:
- Exact tumor volumes (in cm³)
- Number of distinct tumor foci (multifocality)
- Component proportions (enhancing vs non-enhancing tissue)
- Spatial extent of different tumor regions

## Solution: Enriched Text

### What We Added

The enriched text augments original descriptions with:

1. **Volumetric Measurements**
   - Whole Tumor (WT) volume in cm³
   - Tumor Core (TC) volume in cm³
   - Enhancing Tumor (ET) volume in cm³
   - Individual component volumes (edema, necrotic tissue)
   - Relative proportions (e.g., "ET comprises 25% of WT")

2. **Component Counts**
   - Number of distinct tumor foci (unifocal vs multifocal)
   - Number of necrotic regions
   - Number of enhancing lesions
   - Complexity indicators for tumor architecture

3. **Tumor Composition Analysis**
   - Enhancing vs non-enhancing tissue proportions
   - Necrotic tissue extent
   - Perilesional edema extent
   - Heterogeneity characterization

### Implementation

The enriched text is generated using:
- **Volume Analysis**: Computed from segmentation masks using `VolumeCountAnalyzer`
- **AI Integration**: OpenAI GPT-4o or Google Gemini to naturally integrate quantitative data with qualitative descriptions
- **Multimodal Input**: Middle slice images (and optionally videos for Gemini) to provide visual context
- **BioBERT Encoding**: Same embedding approach as original text (1, 128, 768) for compatibility

### File Structure

```
/Disk1/afrouz/Data/Merged/
└── BraTS20_Training_XXX/
    ├── BraTS20_Training_XXX_flair_text.npy          # Original text embedding
    ├── BraTS20_Training_XXX_enriched_text.npy       # New enriched text embedding
    ├── BraTS20_Training_XXX_enriched_text.txt       # Human-readable enriched text
    └── BraTS20_Training_XXX_stats.json              # Volumetric statistics
```

## Why This Matters

### Clinical Relevance
- **Volumetric data** is a standard clinical metric for tumor assessment and treatment planning
- **Multifocality** (tumor count) affects surgical approach and prognosis
- **Component proportions** inform tissue classification and boundary delineation

### Segmentation Benefits
- More precise boundary information from volume data
- Better understanding of tumor heterogeneity from component counts
- Quantitative anchors for the model to learn spatial relationships
- Natural integration with existing qualitative imaging features

## Usage

### Generating Enriched Text

```bash
# Using OpenAI (default)
python utils/enrich_text_with_volume_count.py \
    --data_dir="/Disk1/afrouz/Data/Merged/" \
    --output_dir="/Disk1/afrouz/Data/Merged/" \
    --provider=openai

# Using Gemini with videos
python utils/enrich_text_with_volume_count.py \
    --data_dir="/Disk1/afrouz/Data/Merged/" \
    --output_dir="/Disk1/afrouz/Data/Merged/" \
    --provider=gemini \
    --generate_videos
```

**Note**: The script automatically loads existing enriched text if available, avoiding redundant API calls.

### Training with Enriched Text

Update JSON files to use enriched embeddings:

```bash
python update_text_features.py
```

This updates:
- `Train_extended_text_features.json`
- `Test_extended_text_features.json`

Then train as usual:

```bash
python main.py --use_ssl_pretrained \
    --save_checkpoint --logdir=TextBraTS_enriched_text \
    --data_dir="/Disk1/afrouz/Data/Merged/" \
    --json_list ./Train_extended_text_features.json
```

## Example Comparison

### Original Text
```
High signal intensity on FLAIR in the right frontal lobe with irregular borders
and enhancement pattern suggesting glioblastoma. Necrotic core visible with
surrounding edema.
```

### Enriched Text
```
High signal intensity on FLAIR in the right frontal lobe with irregular borders
and enhancement pattern suggesting glioblastoma. The whole tumor (WT) measures
45.3 cm³, with tumor core (TC) of 28.7 cm³ (63% of WT) and enhancing tumor (ET)
of 15.2 cm³ (34% of WT). The lesion is unifocal with a single necrotic region
comprising 13.5 cm³ (30% of WT), surrounded by 16.6 cm³ of perilesional edema
(37% of WT). The tumor demonstrates moderate heterogeneity with both enhancing
and non-enhancing components, characteristic of high-grade glioma.
```

## Technical Details

### BioBERT Encoding
- Model: `dmis-lab/biobert-base-cased-v1.1`
- Output shape: `(1, 128, 768)` - consistent with original implementation
- Max tokens: 128
- Embedding dimension: 768

### Volume Calculation
- Voxel spacing: (1.5, 1.5, 2.0) mm
- Regions analyzed: NCR/NET (label 1), Edema (label 2), Enhancing (label 4)
- Standard BraTS regions: WT, TC, ET
- Connected component analysis for counting distinct regions

### AI Model Options
- **OpenAI**: GPT-4o (supports images)
- **Gemini**: gemini-2.0-flash-exp (supports images and videos)

## Future Improvements

Potential enhancements:
1. Add shape descriptors (sphericity, compactness)
2. Include texture analysis from imaging
3. Incorporate growth patterns over time (if longitudinal data available)
4. Add relative anatomical position (distance from eloquent structures)

## Related Files

- Implementation: [`utils/enrich_text_with_volume_count.py`](utils/enrich_text_with_volume_count.py)
- JSON updater: [`update_text_features.py`](update_text_features.py)
- Data loader: [`utils/data_utils.py`](utils/data_utils.py)

## References

- BraTS Challenge: https://www.med.upenn.edu/cbica/brats/
- BioBERT: https://github.com/dmis-lab/biobert
- MONAI Framework: https://monai.io/
