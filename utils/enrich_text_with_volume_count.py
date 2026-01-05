"""
Script to enrich TextBraTS descriptions with volume and count features.

This script:
1. Loads FLAIR images and segmentation masks from BraTS dataset
2. Converts 3D volumes to videos (slice-by-slice)
3. Computes volume and count statistics for tumor regions
4. Uses AI API (OpenAI or Gemini) to generate enriched text descriptions
5. Encodes enriched text using BioBERT
"""

import os
import argparse
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Tuple, List
import cv2
from tqdm import tqdm
from scipy import ndimage
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import tempfile
import shutil
import base64


class VolumeCountAnalyzer:
    """Analyzes tumor regions for volume and count statistics."""

    def __init__(self, voxel_spacing=(1.5, 1.5, 2.0)):
        """
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
        """
        self.voxel_spacing = voxel_spacing
        self.voxel_volume = np.prod(voxel_spacing)  # mm^3

    def analyze_segmentation(self, seg_data: np.ndarray) -> Dict:
        """
        Analyze segmentation mask for volume and count features.

        BraTS labels:
        - 0: Background
        - 1: Necrotic/non-enhancing tumor core (NCR/NET)
        - 2: Peritumoral edema (ED)
        - 4: GD-enhancing tumor (ET)

        Standard regions:
        - TC (Tumor Core) = 1 + 4
        - WT (Whole Tumor) = 1 + 2 + 4
        - ET (Enhancing Tumor) = 4

        Args:
            seg_data: 3D segmentation mask

        Returns:
            Dictionary with volume and count statistics
        """
        # Create binary masks for each region
        ncr_net = (seg_data == 1).astype(int)
        edema = (seg_data == 2).astype(int)
        enhancing = (seg_data == 4).astype(int)

        # Standard regions
        tc = np.logical_or(ncr_net, enhancing).astype(int)
        wt = np.logical_or(np.logical_or(ncr_net, edema), enhancing).astype(int)
        et = enhancing

        # Calculate volumes in mm^3 and convert to cm^3
        stats = {
            'ncr_net_volume_cm3': np.sum(ncr_net) * self.voxel_volume / 1000,
            'edema_volume_cm3': np.sum(edema) * self.voxel_volume / 1000,
            'enhancing_volume_cm3': np.sum(enhancing) * self.voxel_volume / 1000,
            'tc_volume_cm3': np.sum(tc) * self.voxel_volume / 1000,
            'wt_volume_cm3': np.sum(wt) * self.voxel_volume / 1000,
            'et_volume_cm3': np.sum(et) * self.voxel_volume / 1000,
        }

        # Count connected components for each region
        stats['ncr_net_count'] = self._count_components(ncr_net)
        stats['edema_count'] = self._count_components(edema)
        stats['enhancing_count'] = self._count_components(enhancing)
        stats['tc_count'] = self._count_components(tc)
        stats['wt_count'] = self._count_components(wt)
        stats['et_count'] = self._count_components(et)

        # Calculate volume ratios
        if stats['wt_volume_cm3'] > 0:
            stats['edema_to_wt_ratio'] = stats['edema_volume_cm3'] / stats['wt_volume_cm3']
            stats['et_to_wt_ratio'] = stats['et_volume_cm3'] / stats['wt_volume_cm3']
            stats['ncr_to_wt_ratio'] = stats['ncr_net_volume_cm3'] / stats['wt_volume_cm3']
        else:
            stats['edema_to_wt_ratio'] = 0.0
            stats['et_to_wt_ratio'] = 0.0
            stats['ncr_to_wt_ratio'] = 0.0

        return stats

    def _count_components(self, binary_mask: np.ndarray) -> int:
        """Count connected components in binary mask."""
        if np.sum(binary_mask) == 0:
            return 0
        labeled, num_components = ndimage.label(binary_mask)
        return num_components


class VideoConverter:
    """Converts 3D medical images to video format."""

    def __init__(self, output_size=(512, 512), fps=5):
        """
        Args:
            output_size: Output frame size (width, height)
            fps: Frames per second for video
        """
        self.output_size = output_size
        self.fps = fps

    def create_video_from_volume(
        self,
        volume: np.ndarray,
        output_path: str,
        normalize: bool = True,
        colormap: str = None
    ):
        """
        Convert 3D volume to video by stacking slices.

        Args:
            volume: 3D numpy array (H, W, D)
            output_path: Path to save video file
            normalize: Whether to normalize intensity
            colormap: OpenCV colormap to apply (None for grayscale)
        """
        # Get dimensions
        height, width, depth = volume.shape

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, self.output_size)

        for slice_idx in range(depth):
            # Extract slice
            slice_2d = volume[:, :, slice_idx]

            # Normalize to 0-255
            if normalize:
                slice_min = slice_2d.min()
                slice_max = slice_2d.max()
                if slice_max > slice_min:
                    slice_2d = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
                else:
                    slice_2d = np.zeros_like(slice_2d, dtype=np.uint8)
            else:
                slice_2d = slice_2d.astype(np.uint8)

            # Apply colormap if specified
            if colormap:
                slice_2d = cv2.applyColorMap(slice_2d, getattr(cv2, f'COLORMAP_{colormap}'))
            else:
                # Convert grayscale to BGR for video
                slice_2d = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2BGR)

            # Resize to output size
            frame = cv2.resize(slice_2d, self.output_size, interpolation=cv2.INTER_LINEAR)

            # Write frame
            out.write(frame)

        out.release()

    def create_overlay_video(
        self,
        flair: np.ndarray,
        seg: np.ndarray,
        output_path: str
    ):
        """
        Create video with segmentation overlay on FLAIR.

        Args:
            flair: 3D FLAIR image
            seg: 3D segmentation mask
            output_path: Path to save video
        """
        height, width, depth = flair.shape

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, self.output_size)

        # Color map for different tumor regions
        colors = {
            1: (255, 0, 0),      # NCR/NET - Red
            2: (0, 255, 0),      # Edema - Green
            4: (0, 0, 255),      # Enhancing - Blue
        }

        for slice_idx in range(depth):
            # Extract slices
            flair_slice = flair[:, :, slice_idx]
            seg_slice = seg[:, :, slice_idx]

            # Normalize FLAIR
            flair_min = flair_slice.min()
            flair_max = flair_slice.max()
            if flair_max > flair_min:
                flair_norm = ((flair_slice - flair_min) / (flair_max - flair_min) * 255).astype(np.uint8)
            else:
                flair_norm = np.zeros_like(flair_slice, dtype=np.uint8)

            # Convert to BGR
            flair_bgr = cv2.cvtColor(flair_norm, cv2.COLOR_GRAY2BGR)

            # Create overlay
            overlay = flair_bgr.copy()
            for label, color in colors.items():
                mask = (seg_slice == label)
                overlay[mask] = np.array(color)

            # Blend
            blended = cv2.addWeighted(flair_bgr, 0.6, overlay, 0.4, 0)

            # Resize
            frame = cv2.resize(blended, self.output_size, interpolation=cv2.INTER_LINEAR)

            # Write frame
            out.write(frame)

        out.release()

    def create_middle_slice_image(
        self,
        flair: np.ndarray,
        seg: np.ndarray,
        output_path: str
    ):
        """
        Create a single image of the middle slice with segmentation overlay on FLAIR.

        Args:
            flair: 3D FLAIR image (H, W, D)
            seg: 3D segmentation mask (H, W, D)
            output_path: Path to save image file
        """
        height, width, depth = flair.shape

        # Get middle slice
        middle_idx = depth // 2
        flair_slice = flair[:, :, middle_idx]
        seg_slice = seg[:, :, middle_idx]

        # Normalize FLAIR
        flair_min = flair_slice.min()
        flair_max = flair_slice.max()
        if flair_max > flair_min:
            flair_norm = ((flair_slice - flair_min) / (flair_max - flair_min) * 255).astype(np.uint8)
        else:
            flair_norm = np.zeros_like(flair_slice, dtype=np.uint8)

        # Convert to BGR
        flair_bgr = cv2.cvtColor(flair_norm, cv2.COLOR_GRAY2BGR)

        # Color map for different tumor regions
        colors = {
            1: (255, 0, 0),      # NCR/NET - Red
            2: (0, 255, 0),      # Edema - Green
            4: (0, 0, 255),      # Enhancing - Blue
        }

        # Create overlay
        overlay = flair_bgr.copy()
        for label, color in colors.items():
            mask = (seg_slice == label)
            overlay[mask] = np.array(color)

        # Blend
        blended = cv2.addWeighted(flair_bgr, 0.6, overlay, 0.4, 0)

        # Resize to output size
        image = cv2.resize(blended, self.output_size, interpolation=cv2.INTER_LINEAR)

        # Save image
        cv2.imwrite(output_path, image)


class TextEnricher:
    """Uses AI API (Gemini or OpenAI) via LangChain to generate enriched text descriptions."""

    def __init__(self, api_key: str, provider: str = "gemini", model_name: str = None):
        """
        Args:
            api_key: API key for the chosen provider
            provider: Either "gemini" or "openai"
            model_name: Model to use (defaults to provider-specific models)
        """
        self.provider = provider.lower()

        if self.provider == "gemini":
            if model_name is None:
                model_name = "gemini-2.0-flash-exp"
            self.model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.7
            )
        elif self.provider == "openai":
            if model_name is None:
                model_name = "gpt-4o"
            self.model = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                temperature=0.7
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'gemini' or 'openai'.")

        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a medical imaging expert analyzing brain MRI scans for glioma segmentation."),
            ("human", """{prompt_content}""")
        ])

    def enrich_text(
        self,
        original_text: str,
        stats: Dict,
        flair_video_path: str = None,
        seg_video_path: str = None,
        middle_slice_path: str = None
    ) -> str:
        """
        Enrich text description with volume and count features.

        Args:
            original_text: Original text description
            stats: Volume and count statistics
            flair_video_path: Path to FLAIR video (optional, Gemini only)
            seg_video_path: Path to segmentation video (optional, Gemini only)
            middle_slice_path: Path to middle slice image with overlay (optional, for both providers)

        Returns:
            Enriched text description
        """
        # Format statistics for the prompt
        stats_text = self._format_statistics(stats)

        # Create prompt content
        prompt_content = f"""Original Description:
{original_text}

Quantitative Measurements:
{stats_text}

Task: Generate a clinically-oriented description that integrates quantitative tumor measurements with qualitative imaging characteristics. Focus on features relevant for segmentation and clinical assessment.

Content Requirements:
1. **Tumor Volumetrics**: Report volumes of WT, TC, and ET in cm³, including their relative proportions (e.g., "ET comprises 25% of WT").

2. **Tumor Composition**: Describe the makeup of the tumor based on component volumes:
   - Enhancing vs non-enhancing proportions
   - Necrotic tissue extent
   - Perilesional edema extent

3. **Heterogeneity & Multifocality**: Use component counts to characterize:
   - Number of distinct tumor foci (unifocal vs multifocal)
   - Number of necrotic regions (indicating heterogeneity)
   - Complexity of tumor architecture

4. **Imaging Characteristics**: Preserve key qualitative features from original:
   - Signal intensity patterns (high/low/mixed)
   - Enhancement characteristics
   - Border characteristics (irregular, infiltrative, well-defined)
   - Necrotic vs viable tissue patterns

Guidelines:
- Integrate numbers naturally into medical descriptions
- Prioritize features that inform tissue classification and boundary delineation

Enriched Description:"""

        # Generate content with multimodal input (images/videos) or text-only
        media_content = []

        # Add middle slice image if provided (supported by both OpenAI and Gemini)
        if middle_slice_path and os.path.exists(middle_slice_path):
            try:
                with open(middle_slice_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    if self.provider == "gemini":
                        media_content.append({
                            "type": "media",
                            "mime_type": "image/png",
                            "data": image_data
                        })
                    elif self.provider == "openai":
                        media_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        })
            except Exception as e:
                print(f"Warning: Failed to process middle slice image: {e}")

        # Add videos for Gemini (if provided)
        if self.provider == "gemini" and flair_video_path and os.path.exists(flair_video_path):
            try:
                # Read and encode FLAIR video
                with open(flair_video_path, 'rb') as f:
                    flair_data = base64.b64encode(f.read()).decode('utf-8')
                    media_content.append({
                        "type": "media",
                        "mime_type": "video/mp4",
                        "data": flair_data
                    })

                # Read and encode segmentation overlay video if it exists
                if seg_video_path and os.path.exists(seg_video_path):
                    with open(seg_video_path, 'rb') as f:
                        seg_data = base64.b64encode(f.read()).decode('utf-8')
                        media_content.append({
                            "type": "media",
                            "mime_type": "video/mp4",
                            "data": seg_data
                        })
            except Exception as e:
                print(f"Warning: Failed to process videos: {e}")

        # Generate response
        if media_content:
            try:
                # Create message with multimodal content
                message = HumanMessage(
                    content=[{"type": "text", "text": prompt_content}] + media_content
                )
                response = self.model.invoke([message])
            except Exception as e:
                print(f"Warning: Failed to process media content, falling back to text-only: {e}")
                # Fallback to text-only if media processing fails
                chain = self.prompt_template | self.model
                response = chain.invoke({"prompt_content": prompt_content})
        else:
            # Text-only generation
            chain = self.prompt_template | self.model
            response = chain.invoke({"prompt_content": prompt_content})

        return response.content.strip()

    def _format_statistics(self, stats: Dict) -> str:
        """Format statistics dictionary into readable text."""
        lines = []

        # Volume information
        lines.append("Volume Measurements (cm³):")
        lines.append(f"  - Whole Tumor (WT): {stats['wt_volume_cm3']:.2f}")
        lines.append(f"  - Tumor Core (TC): {stats['tc_volume_cm3']:.2f}")
        lines.append(f"  - Enhancing Tumor (ET): {stats['et_volume_cm3']:.2f}")
        lines.append(f"  - Edema: {stats['edema_volume_cm3']:.2f}")
        lines.append(f"  - Necrotic/Non-enhancing: {stats['ncr_net_volume_cm3']:.2f}")

        # Component counts
        lines.append("\nComponent Counts:")
        lines.append(f"  - Whole Tumor lesions: {stats['wt_count']}")
        lines.append(f"  - Tumor Core lesions: {stats['tc_count']}")
        lines.append(f"  - Enhancing lesions: {stats['et_count']}")
        lines.append(f"  - Edema regions: {stats['edema_count']}")
        lines.append(f"  - Necrotic regions: {stats['ncr_net_count']}")

        # Ratios
        lines.append("\nVolume Ratios:")
        lines.append(f"  - Edema/WT: {stats['edema_to_wt_ratio']:.2%}")
        lines.append(f"  - ET/WT: {stats['et_to_wt_ratio']:.2%}")
        lines.append(f"  - Necrotic/WT: {stats['ncr_to_wt_ratio']:.2%}")

        return "\n".join(lines)


class BioBERTEncoder:
    """Encodes text using BioBERT."""

    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", device: str = None):
        """
        Args:
            model_name: BioBERT model name from HuggingFace
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Force use of safetensors to avoid torch.load vulnerability
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        ).to(device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text using BioBERT.

        Args:
            text: Input text

        Returns:
            Text embedding as numpy array of shape (1, 128, 768)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        ).to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use first 128 tokens (including [CLS]), each with 768 dimensions
            # Shape: (1, 128, 768) - keep the original shape to match existing embeddings
            embeddings = outputs.last_hidden_state[:, :128, :].cpu().numpy()

        return embeddings


def process_single_case(
    case_dir: Path,
    output_dir: Path,
    analyzer: VolumeCountAnalyzer,
    video_converter: VideoConverter,
    text_enricher: TextEnricher,
    biobert_encoder: BioBERTEncoder,
    generate_videos: bool = True
) -> Dict:
    """
    Process a single BraTS case.

    Args:
        case_dir: Path to case directory
        output_dir: Path to output directory
        analyzer: VolumeCountAnalyzer instance
        video_converter: VideoConverter instance
        text_enricher: TextEnricher instance
        biobert_encoder: BioBERTEncoder instance
        generate_videos: Whether to generate videos for AI analysis

    Returns:
        Dictionary with processing results
    """
    case_name = case_dir.name

    # Load FLAIR and segmentation
    flair_path = case_dir / f"{case_name}_flair.nii"
    seg_path = case_dir / f"{case_name}_seg.nii"
    text_path = case_dir / f"{case_name}_flair_text.txt"

    if not flair_path.exists() or not seg_path.exists():
        return {'status': 'error', 'message': 'Missing FLAIR or segmentation file'}

    # Load data
    flair_nii = nib.load(str(flair_path))
    flair_data = flair_nii.get_fdata()

    seg_nii = nib.load(str(seg_path))
    seg_data = seg_nii.get_fdata().astype(np.uint8)

    # Load original text if exists
    if text_path.exists():
        with open(text_path, 'r') as f:
            original_text = f.read().strip()
    else:
        original_text = "No original description available."

    # Analyze volume and count
    stats = analyzer.analyze_segmentation(seg_data)

    # Generate visual content (middle slice and optional videos)
    temp_dir = tempfile.mkdtemp()
    flair_video_path = None
    overlay_video_path = None
    middle_slice_path = None

    try:
        # Always create middle slice image (works with both OpenAI and Gemini)
        middle_slice_path = os.path.join(temp_dir, f"{case_name}_middle_slice.png")
        video_converter.create_middle_slice_image(flair_data, seg_data, middle_slice_path)

        # Generate videos if requested (Gemini only)
        if generate_videos:
            flair_video_path = os.path.join(temp_dir, f"{case_name}_flair.mp4")
            overlay_video_path = os.path.join(temp_dir, f"{case_name}_overlay.mp4")

            video_converter.create_video_from_volume(flair_data, flair_video_path)
            video_converter.create_overlay_video(flair_data, seg_data, overlay_video_path)

        # Enrich text with AI (with middle slice and optional videos)
        enriched_text = text_enricher.enrich_text(
            original_text,
            stats,
            flair_video_path,
            overlay_video_path,
            middle_slice_path
        )
    except Exception as e:
        print(f"Error generating media or calling AI for {case_name}: {e}")
        # Fallback to text-only if media generation fails
        enriched_text = text_enricher.enrich_text(original_text, stats)
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Encode with BioBERT
    text_embedding = biobert_encoder.encode(enriched_text)

    # Save outputs
    case_output_dir = output_dir / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Save enriched text
    enriched_text_path = case_output_dir / f"{case_name}_enriched_text.txt"
    with open(enriched_text_path, 'w') as f:
        f.write(enriched_text)

    # Save embedding
    embedding_path = case_output_dir / f"{case_name}_enriched_text.npy"
    np.save(embedding_path, text_embedding)

    # Save statistics
    stats_path = case_output_dir / f"{case_name}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return {
        'status': 'success',
        'case_name': case_name,
        'original_text': original_text,
        'enriched_text': enriched_text,
        'stats': stats,
        'embedding_shape': text_embedding.shape
    }


def main():
    parser = argparse.ArgumentParser(description="Enrich TextBraTS with volume and count features")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Disk1/afrouz/Data/Merged',
        help='Path to BraTS data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/Disk1/afrouz/Data/Merged_Enriched',
        help='Path to output directory'
    )
    parser.add_argument(
        '--biobert_model',
        type=str,
        default='dmis-lab/biobert-base-cased-v1.1',
        help='BioBERT model name'
    )
    parser.add_argument(
        '--generate_videos',
        action='store_true',
        help='Generate videos for AI analysis (only supported with Gemini provider)'
    )
    parser.add_argument(
        '--num_cases',
        type=int,
        default=0,
        help='Number of cases to process (use 0 or negative for all cases)'
    )
    parser.add_argument(
        '--voxel_spacing',
        type=float,
        nargs=3,
        default=[1.5, 1.5, 2.0],
        help='Voxel spacing in mm (x y z)'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['gemini', 'openai'],
        default='openai',
        help='AI provider to use for text enrichment (gemini or openai)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Specific model name to use (default: gpt-4o for OpenAI, gemini-2.0-flash-exp for Gemini)'
    )

    args = parser.parse_args()

    # Get API key from environment variable based on provider
    if args.provider == 'gemini':
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it using: export GEMINI_API_KEY='your-api-key'"
            )
    elif args.provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it using: export OPENAI_API_KEY='your-api-key'"
            )
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    print("Initializing components...")
    print(f"Using {args.provider} for text enrichment...")
    analyzer = VolumeCountAnalyzer(voxel_spacing=tuple(args.voxel_spacing))
    video_converter = VideoConverter()
    text_enricher = TextEnricher(
        api_key=api_key,
        provider=args.provider,
        model_name=args.model_name
    )
    biobert_encoder = BioBERTEncoder(model_name=args.biobert_model)

    # Get case directories
    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('BraTS')])

    # Limit number of cases (if num_cases > 0, otherwise process all)
    if args.num_cases > 0:
        case_dirs = case_dirs[:args.num_cases]

    print(f"Processing {len(case_dirs)} cases...")

    # Process cases
    results = []
    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        try:
            result = process_single_case(
                case_dir,
                output_dir,
                analyzer,
                video_converter,
                text_enricher,
                biobert_encoder,
                args.generate_videos
            )
            results.append(result)

            if result['status'] == 'success':
                print(f"\n{'='*80}")
                print(f"Case: {result['case_name']}")
                print(f"\nOriginal Text:\n{result['original_text']}")
                print(f"\nEnriched Text:\n{result['enriched_text']}")
                print(f"\nKey Stats:")
                print(f"  WT Volume: {result['stats']['wt_volume_cm3']:.2f} cm³ ({result['stats']['wt_count']} lesion(s))")
                print(f"  TC Volume: {result['stats']['tc_volume_cm3']:.2f} cm³ ({result['stats']['tc_count']} lesion(s))")
                print(f"  ET Volume: {result['stats']['et_volume_cm3']:.2f} cm³ ({result['stats']['et_count']} lesion(s))")
                print(f"{'='*80}\n")
            else:
                print(f"Error processing {case_dir.name}: {result['message']}")

        except Exception as e:
            print(f"Error processing {case_dir.name}: {e}")
            results.append({'status': 'error', 'case_name': case_dir.name, 'message': str(e)})

    # Save summary
    summary_path = output_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"Successfully processed: {success_count}/{len(results)} cases")
    print(f"Output directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

