"""
Combined Text-Constrained Loss for BraTS Segmentation

Integrates three complementary loss components:
1. Dice Loss (base segmentation objective)
2. Volumetric Constraint Loss (enforce size matching from text)
3. Spatial Constraint Loss (enforce anatomical location matching from text)

Based on methodology from:
- Kervadec et al. (2019) "Constrained-CNN losses for weakly supervised segmentation"
- TextBraTS: Text-Guided Volumetric Brain Tumor Segmentation

Author: TextBraTS Project
Date: 2025-12-05
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple


class TextConstrainedLoss(nn.Module):
    """
    Complete text-constrained loss combining base segmentation loss with text-guided constraints

    Loss formulation:
        L_total = L_dice + λ_vol * L_volumetric + λ_spatial * L_spatial

    where:
        - L_dice: Base Dice loss for segmentation
        - L_volumetric: Volumetric constraint loss (HIGH/MODERATE/LOW matching)
        - L_spatial: Spatial constraint loss (anatomical region containment)
        - λ_vol, λ_spatial: Weighting hyperparameters
    """

    def __init__(self,
                 dice_loss: nn.Module,
                 volumetric_loss: Optional[nn.Module] = None,
                 spatial_loss: Optional[nn.Module] = None,
                 volume_weight: float = 0.1,
                 spatial_weight: float = 0.1,
                 return_loss_dict: bool = False):
        """
        Args:
            dice_loss: Base DiceLoss module (e.g., from MONAI)
            volumetric_loss: VolumetricConstraintLoss module (optional)
            spatial_loss: SpatialConstraintLoss module (optional)
            volume_weight: λ_vol weight for volumetric term
            spatial_weight: λ_spatial weight for spatial term
            return_loss_dict: If True, forward() returns (loss, dict), else just loss
        """
        super().__init__()

        self.dice_loss = dice_loss
        self.vol_loss = volumetric_loss
        self.spatial_loss = spatial_loss

        self.lambda_vol = volume_weight
        self.lambda_spatial = spatial_weight

        # Flags for which losses are active
        self.use_vol = volumetric_loss is not None
        self.use_spatial = spatial_loss is not None

        self.return_dict = return_loss_dict

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                sample_ids: Optional[List[str]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute combined loss

        Args:
            logits: (B, 3, H, W, D) raw model outputs before sigmoid
            targets: (B, 3, H, W, D) ground truth masks
            sample_ids: List[str] sample identifiers (required for constraint losses)

        Returns:
            If return_loss_dict is False:
                total_loss: Scalar loss value
            If return_loss_dict is True:
                (total_loss, loss_dict): Loss and dictionary with components
        """
        # 1. Dice loss (always computed)
        loss_dice = self.dice_loss(logits, targets)
        total_loss = loss_dice

        # Initialize loss dictionary
        loss_dict = {'dice': loss_dice.item()}

        # 2. Volumetric constraint loss
        if self.use_vol and sample_ids is not None:
            loss_vol = self.vol_loss(logits, sample_ids)
            total_loss = total_loss + self.lambda_vol * loss_vol
            loss_dict['volumetric'] = loss_vol.item()
        else:
            loss_dict['volumetric'] = 0.0

        # 3. Spatial constraint loss
        if self.use_spatial and sample_ids is not None:
            loss_spatial = self.spatial_loss(logits, sample_ids)
            total_loss = total_loss + self.lambda_spatial * loss_spatial
            loss_dict['spatial'] = loss_spatial.item()
        else:
            loss_dict['spatial'] = 0.0

        # Store total loss
        loss_dict['total'] = total_loss.item()

        # Return based on configuration
        if self.return_dict:
            return total_loss, loss_dict
        else:
            return total_loss

    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return {
            'dice': 1.0,
            'volumetric': self.lambda_vol if self.use_vol else 0.0,
            'spatial': self.lambda_spatial if self.use_spatial else 0.0
        }

    def set_loss_weights(self, volume_weight: Optional[float] = None,
                        spatial_weight: Optional[float] = None):
        """
        Update loss weights (useful for curriculum learning)

        Args:
            volume_weight: New λ_vol (if provided)
            spatial_weight: New λ_spatial (if provided)
        """
        if volume_weight is not None:
            self.lambda_vol = volume_weight
        if spatial_weight is not None:
            self.lambda_spatial = spatial_weight


def create_text_constrained_loss(args, dice_loss: nn.Module) -> nn.Module:
    """
    Factory function to create text-constrained loss based on configuration

    This function handles initialization of all constraint losses based on
    command-line arguments or configuration.

    Args:
        args: Training arguments/configuration with fields:
            - use_volumetric_constraint: bool
            - use_spatial_constraint: bool
            - volumetric_json_path: str
            - atlas_path: str (if spatial loss enabled)
            - volume_weight: float
            - spatial_weight: float
            - data_driven_thresholds: bool (optional)
            - data_dir: str (if data_driven_thresholds=True)
        dice_loss: Pre-initialized DiceLoss module

    Returns:
        loss_func: TextConstrainedLoss module or DiceLoss (if no constraints)
    """
    volumetric_loss = None
    spatial_loss = None

    # Create volumetric constraint loss if enabled
    if args.use_volumetric_constraint:
        from losses.volumetric_constraint_loss import (
            VolumetricConstraintLoss,
            VOLUME_THRESHOLDS_LITERATURE
        )

        # Get volume thresholds
        if hasattr(args, 'data_driven_thresholds') and args.data_driven_thresholds:
            # Import threshold calibration utility
            from losses.compute_volume_thresholds import compute_volume_thresholds
            thresholds = compute_volume_thresholds(args.data_dir)
            print("Using data-driven volume thresholds")
        else:
            thresholds = VOLUME_THRESHOLDS_LITERATURE
            print("Using literature-based volume thresholds")

        volumetric_loss = VolumetricConstraintLoss(
            json_path=args.volumetric_json_path,
            thresholds=thresholds,
            use_normalized_volumes=True
        )

        print(f"Volumetric constraint loss enabled (λ_vol={args.volume_weight})")

    # Create spatial constraint loss if enabled
    if args.use_spatial_constraint:
        from losses.spatial_constraint_loss import SpatialConstraintLoss

        spatial_loss = SpatialConstraintLoss(
            json_path=args.volumetric_json_path,
            atlas_path=args.atlas_path
        )

        print(f"Spatial constraint loss enabled (λ_spatial={args.spatial_weight})")

    # Create combined loss if any constraints are active
    if volumetric_loss is not None or spatial_loss is not None:
        loss_func = TextConstrainedLoss(
            dice_loss=dice_loss,
            volumetric_loss=volumetric_loss,
            spatial_loss=spatial_loss,
            volume_weight=args.volume_weight,
            spatial_weight=args.spatial_weight,
            return_loss_dict=True  # Return dict for detailed logging
        )

        print("Using TextConstrainedLoss with:")
        if volumetric_loss:
            print(f"  ✓ Volumetric constraint (weight={args.volume_weight})")
        if spatial_loss:
            print(f"  ✓ Spatial constraint (weight={args.spatial_weight})")

        return loss_func
    else:
        # No constraints - return base Dice loss
        print("Using DiceLoss only (no text constraints)")
        return dice_loss
