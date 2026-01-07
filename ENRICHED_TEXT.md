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

## Experimental Observations

### Performance Analysis: Original vs Enriched Text

#### Key Findings

After training and testing models with both original text (flair_text.npy) and enriched text (enriched_text.npy), we observed an interesting pattern:

**High-Performing Cases (90%+ Dice Score)**
- Original text: Achieves 90%+ dice scores
- Enriched text: Performance **degrades** to ~85% dice
- **Hypothesis**: Simple, well-defined tumors with clear boundaries don't benefit from quantitative data. The volumetric information may introduce noise/redundancy that confuses the model, causing it to over-fit to volume numbers rather than spatial features.

**Lower-Performing Cases (<90% Dice Score)**
- Original text: Struggles with accurate segmentation
- Enriched text: Shows **improved** performance
- **Hypothesis**: Complex, ambiguous, or multifocal tumors benefit from quantitative constraints. Volume measurements and component counts provide critical information that qualitative descriptions lack.

#### Interpretation

This suggests:
1. **Information Overload**: For simple cases, enriched text adds unnecessary complexity
2. **Critical Augmentation**: For complex cases, quantitative data fills gaps in qualitative descriptions
3. **Case-Dependent Value**: The benefit of enriched text is not uniform across the dataset

#### Implications for Model Development

These observations suggest that **simple concatenation or averaging** of embeddings may be suboptimal. Instead, consider:

1. **Adaptive Fusion**: Model learns to weight embeddings based on tumor complexity
   ```python
   gate = sigmoid(learned_function(image_features))
   combined = gate * enriched_embedding + (1 - gate) * flair_embedding
   ```

2. **Separate Pathways**: Process each embedding independently, combine predictions
   ```python
   pred_flair = model_branch(flair_embedding, image)
   pred_enriched = model_branch(enriched_embedding, image)
   final_pred = learned_weight * pred_flair + (1-learned_weight) * pred_enriched
   ```

3. **Conditional Routing**: Detect tumor complexity and select appropriate embedding
   - Use enriched text for large, multifocal, heterogeneous tumors
   - Use original text for small, unifocal, well-defined tumors

#### Next Steps

1. **Quantify the pattern**: Analyze correlation between tumor characteristics (size, multifocality, heterogeneity) and performance delta
2. **Implement adaptive fusion**: Let model learn when to trust each embedding type
3. **Visualize cases**: Compare samples where each approach excels to identify distinguishing features

#### Related Outputs
- Original text results: `/Disk1/afrouz/Projects/TextBraTS/outputs/TextBraTS_conda_viz/test_visualizations.pdf`
- Enriched text results: `/Disk1/afrouz/Projects/TextBraTS/outputs/TextBraTS_conda_enriched_text/test_visualizations.pdf`

## Late Fusion Implementation

### Overview

To address the observation that enriched text helps some cases but hurts others, we implemented **late fusion with learnable weights**.

### How It Works

The model performs two forward passes and fuses predictions:
```python
final_prediction = alpha * flair_prediction + (1 - alpha) * enriched_prediction
```

Where `alpha` is a learnable parameter (initialized to 0.5) that the model optimizes during training.

### Files Created/Modified

1. **Data Preparation**:
   - `Train_dual_text_features.json` - Training set with both text features
   - `Test_dual_text_features.json` - Test set with both text features
   - `create_dual_text_json.py` - Script to merge flair and enriched JSON files

2. **Model**:
   - `utils/late_fusion_wrapper.py` - LateFusionWrapper class
   - Wraps any TextBraTS model
   - Implements learnable alpha parameter (clamped to [0, 1])

3. **Training/Testing**:
   - `main.py` - Added `--use_dual_text` argument
   - `trainer.py` - Updated to handle dual text embeddings
   - `test.py` - Updated to handle dual text embeddings
   - `utils/data_utils.py` - Updated data loader

### Usage

**Training:**
```bash
python main.py \
    --use_ssl_pretrained \
    --save_checkpoint \
    --logdir=TextBraTS_late_fusion \
    --data_dir="/Disk1/afrouz/Data/Merged/" \
    --json_list=./Train_dual_text_features.json \
    --use_dual_text \
    --max_epochs=200
```

**Testing:**
```bash
python test.py \
    --data_dir="/Disk1/afrouz/Data/Merged/" \
    --json_list=./Test_dual_text_features.json \
    --pretrained_dir="./runs/TextBraTS_late_fusion/" \
    --use_dual_text \
    --visualize
```

### Expected Benefits

- Model learns optimal global weighting between flair and enriched text
- Single learnable parameter (alpha) adapts to dataset characteristics
- Computational cost: ~2x (dual forward passes)

## References

- BraTS Challenge: https://www.med.upenn.edu/cbica/brats/
- BioBERT: https://github.com/dmis-lab/biobert
- MONAI Framework: https://monai.io/
