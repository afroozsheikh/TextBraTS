import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGuidedRankingLoss(nn.Module):
    def __init__(self, basic_margin=100.0):
        super().__init__()
        # basic_margin: A minimum voxel difference to enforce stability.
        # e.g., if Edema > Core, it should be bigger by at least 'basic_margin' voxels.
        self.basic_margin = basic_margin

    def forward(self, preds, dominance_scores):
        """
        preds: (B, 3, H, W, D) - Sigmoid outputs for [WT, TC, ET]
        dominance_scores: (B,) - Float tensor from LLM (-1.0 to 1.0)
                                 >0 implies Edema > Core
                                 <0 implies Core > Edema
        """
        # 1. Soft Volume Calculation
        # WT = Channel 0, TC = Channel 1
        # Edema volume approximation = WT - TC
        # We clamp to 0 just in case the model predicts TC > WT (which is anatomically impossible but can happen during training)
        
        prob_wt = preds[:, 0, ...]
        prob_tc = preds[:, 1, ...]
        
        # Soft sum over spatial dimensions (H, W, D)
        vol_wt = torch.sum(prob_wt, dim=[1, 2, 3])
        vol_tc = torch.sum(prob_tc, dim=[1, 2, 3])
        
        vol_edema = vol_wt - vol_tc
        vol_core = vol_tc

        # 2. Calculate the "Difference" vector
        # diff > 0 means Edema is bigger. diff < 0 means Core is bigger.
        diff = vol_edema - vol_core
        
        # 3. Dynamic Margin Ranking Loss
        # We want: sign(diff) == sign(dominance_score)
        
        loss = 0.0
        # Iterate simply to handle the conditional logic clearly (vectorize later for speed if needed)
        for i in range(len(dominance_scores)):
            score = dominance_scores[i]
            current_diff = diff[i]
            
            if score > 0: 
                # Case: Text says "Significant Edema" -> Expect Edema > Core
                # We punish if (Edema - Core) is small or negative
                # Target: current_diff should be positive and large
                
                # Using the score to scale the margin. 
                # If score is 1.0 (Massive), margin is larger.
                target_margin = self.basic_margin * score 
                
                # Loss = ReLU(Target - Actual)
                # If current_diff is 500 and target is 100, Loss is 0.
                # If current_diff is -200 (Core is bigger), Loss is High.
                loss += F.relu(target_margin - current_diff)

            elif score < 0:
                # Case: Text says "Minor Edema" -> Expect Core > Edema
                # We punish if (Edema - Core) is positive
                # Target: current_diff should be negative
                
                # score is negative, so we flip logic
                target_margin = self.basic_margin * abs(score)
                
                # We want current_diff to be less than -target_margin
                # i.e., Core should be bigger than Edema by margin
                loss += F.relu(current_diff + target_margin)

        return loss / len(dominance_scores)