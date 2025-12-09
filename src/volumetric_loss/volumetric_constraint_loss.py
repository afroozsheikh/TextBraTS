"""
Volumetric Constraint Loss for Text-Guided BraTS Segmentation

Implements inequality constraint losses based on:
Kervadec et al. (2019) "Constrained-CNN losses for weakly supervised segmentation"
Medical Image Analysis, 54, 88-99. arXiv:1805.04628

Author: TextBraTS Project
Date: 2025-12-05
"""

import torch
import torch.nn as nn
import json


# Literature-based volume thresholds (normalized by image volume)
# Based on typical BraTS tumor size distributions
VOLUME_THRESHOLDS_LITERATURE = {
    'WT': {
        'LOW': (0, 0.005),      # < 0.5% of brain volume
        'MODERATE': (0.005, 0.02),  # 0.5% - 2%
        'HIGH': (0.02, 1.0)         # > 2%
    },
    'TC': {
        'LOW': (0, 0.003),
        'MODERATE': (0.003, 0.015),
        'HIGH': (0.015, 1.0)
    },
    'ET': {
        'LOW': (0, 0.001),
        'MODERATE': (0.001, 0.008),
        'HIGH': (0.008, 1.0)
    }
}


class VolumetricConstraintLoss(nn.Module):
    """
    Enforces volume constraints from text labels (HIGH/MODERATE/LOW)

    Mathematical formulation:
        L_volumetric = Σ_c [ ReLU(V_pred^c - V_max^c) + ReLU(V_min^c - V_pred^c) ]

    where:
        V_pred^c = Σ_{i,j,k} P^c_{i,j,k}  (soft volume count)
        P^c = sigmoid(logits^c)  (predicted probabilities)
        c ∈ {TC, WT, ET}
    """

    def __init__(self,
                 json_path: str,
                 thresholds: dict,
                 use_normalized_volumes: bool = True):
        """
        Args:
            json_path: Path to volumetric_extractions.json
            thresholds: Dict mapping channel -> extent -> (min, max) voxels
                       Format: {'WT': {'LOW': (min, max), 'MODERATE': ..., 'HIGH': ...}, ...}
            use_normalized_volumes: If True, normalize volumes by total image size
        """
        super().__init__()

        # Load JSON constraints
        with open(json_path, 'r') as f:
            self.constraints_db = json.load(f)

        self.thresholds = thresholds
        self.use_normalized = use_normalized_volumes

        # Mapping from JSON pathology to BraTS channel index
        # BraTS channels: [TC, WT, ET] = [0, 1, 2]
        # TC (Tumor Core) = Lesion + Necrosis
        # WT (Whole Tumor) = Lesion + Edema + Necrosis
        # ET (Enhancing Tumor) = Active tumor component
        self.pathology_to_channel = {
            'Lesion': 1,  # Use WT as proxy for overall lesion
            'Edema': 1,   # WT contains edema
            'Necrosis': 0  # TC contains necrosis
        }

    def get_volume(self, probs, channel_idx):
        """
        Compute soft volume for a channel

        Args:
            probs: (B, 3, H, W, D) sigmoid probabilities
            channel_idx: Which channel to compute (0=TC, 1=WT, 2=ET)

        Returns:
            volumes: (B,) tensor of soft voxel counts
        """
        channel_probs = probs[:, channel_idx, ...]  # (B, H, W, D)
        volumes = torch.sum(channel_probs, dim=(1, 2, 3))  # (B,)

        if self.use_normalized:
            total_voxels = channel_probs.shape[1] * channel_probs.shape[2] * channel_probs.shape[3]
            volumes = volumes / total_voxels

        return volumes

    def forward(self, logits, sample_ids):
        """
        Compute volumetric constraint violation penalty

        Args:
            logits: (B, 3, H, W, D) raw model outputs before sigmoid
            sample_ids: List[str] like ["BraTS20_Training_001", ...]

        Returns:
            loss: Scalar volumetric constraint violation penalty
        """
        if sample_ids is None:
            return torch.tensor(0.0, device=logits.device)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        batch_size = probs.shape[0]
        total_loss = 0.0
        valid_samples = 0

        for b in range(batch_size):
            sample_id = sample_ids[b]

            # Skip if no constraints for this sample
            if sample_id not in self.constraints_db:
                continue

            sample_data = self.constraints_db[sample_id]
            overall_burden = sample_data.get('Overall_Burden', {})

            # Process each pathology type
            for pathology, channel_idx in self.pathology_to_channel.items():
                extent_key = f"{pathology}_Extent"
                extent = overall_burden.get(extent_key, None)

                if extent is None or extent == "NONE":
                    continue

                # Get predicted volume for this channel
                pred_vol = self.get_volume(probs[b:b+1], channel_idx)

                # Get threshold range for this extent
                channel_name = ['TC', 'WT', 'ET'][channel_idx]

                # Handle case where extent is not in thresholds
                if extent not in self.thresholds[channel_name]:
                    continue

                v_min, v_max = self.thresholds[channel_name][extent]

                # Compute violation penalties (ReLU-based margin loss)
                # Penalty if volume is too small
                loss_min = torch.relu(v_min - pred_vol)
                # Penalty if volume is too large
                loss_max = torch.relu(pred_vol - v_max)

                total_loss += (loss_min + loss_max)
                valid_samples += 1

        # Average over valid samples (avoid division by zero)
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=logits.device)


class DiceVolumetricLoss(nn.Module):
    """
    Combined Dice + Volumetric constraint loss

    L_total = L_dice + λ_vol * L_volumetric
    """

    def __init__(self,
                 dice_loss,
                 volumetric_loss,
                 volume_weight: float = 0.1):
        """
        Args:
            dice_loss: MONAI DiceLoss module
            volumetric_loss: VolumetricConstraintLoss module
            volume_weight: λ_vol weight for volumetric term
        """
        super().__init__()
        self.dice_loss = dice_loss
        self.vol_loss = volumetric_loss
        self.lambda_vol = volume_weight

    def forward(self, logits, targets, sample_ids=None):
        """
        Args:
            logits: (B, 3, H, W, D) raw model outputs
            targets: (B, 3, H, W, D) ground truth masks
            sample_ids: List[str] or None

        Returns:
            total_loss: Combined loss value
        """
        # Dice loss (always computed)
        loss_dice = self.dice_loss(logits, targets)

        # Volumetric constraint loss (only if sample_ids provided)
        if sample_ids is not None:
            loss_vol = self.vol_loss(logits, sample_ids)
            total_loss = loss_dice + self.lambda_vol * loss_vol
        else:
            total_loss = loss_dice

        return total_loss
