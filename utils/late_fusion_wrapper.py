"""
Late Fusion Wrapper for TextBraTS
Combines predictions from dual text embeddings (flair_text + enriched_text)
with learnable fusion weights.
"""

import torch
import torch.nn as nn


class LateFusionWrapper(nn.Module):
    """
    Wrapper that performs late fusion of dual text embeddings with learnable weights.

    Strategy:
    1. Forward pass with flair_text embedding → logits_flair
    2. Forward pass with enriched_text embedding → logits_enriched
    3. Fuse: logits_final = alpha * logits_flair + (1 - alpha) * logits_enriched
       where alpha is learned during training

    Args:
        base_model: The underlying TextSwinUNETR model
        initial_alpha: Initial value for the fusion weight (default: 0.5)
    """

    def __init__(self, base_model, initial_alpha=0.5):
        super().__init__()
        self.base_model = base_model

        # Learnable fusion weight (initialized to initial_alpha)
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))

    def forward(self, x_in, text_flair, text_enriched):
        """
        Forward pass with dual text embeddings.

        Args:
            x_in: Input image (B, C, H, W, D)
            text_flair: Original flair text embedding (B, 128, 768)
            text_enriched: Enriched text embedding (B, 128, 768)

        Returns:
            logits_final: Fused segmentation logits (B, num_classes, H, W, D)
        """
        # Forward pass with flair text
        logits_flair = self.base_model(x_in, text_flair)

        # Forward pass with enriched text
        logits_enriched = self.base_model(x_in, text_enriched)

        # Late fusion with learnable weight
        # Clamp alpha to [0, 1] for stability
        alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
        logits_final = alpha_clamped * logits_flair + (1 - alpha_clamped) * logits_enriched

        return logits_final

    def get_fusion_weight(self):
        """Return the current fusion weight (alpha)."""
        return torch.clamp(self.alpha, 0.0, 1.0).item()

    def __repr__(self):
        alpha_val = self.get_fusion_weight()
        return (f"LateFusionWrapper(\n"
                f"  base_model={self.base_model.__class__.__name__},\n"
                f"  alpha={alpha_val:.4f} (learnable)\n"
                f"  fusion: {alpha_val:.2%} flair + {(1-alpha_val):.2%} enriched\n"
                f")")
