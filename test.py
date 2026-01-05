# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from utils.data_utils import get_loader
from utils.textswin_unetr import TextSwinUNETR
from utils.textswin_unetr_swinUnetR_V2 import TextSwinUNETR_V2
from utils.textswin_unetr_tf_per_decoder import TextSwinUNETRTFDecoder
from utils.textswin_unetr_jf_per_decoder import TextSwinUNETRJFDecoder
import os
import time
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from utils.utils import AverageMeter
from monai.utils.enums import MetricReduction
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def create_visualization_pdf(viz_data, output_path):
    """
    Create a PDF with visualizations of all test samples.
    Each page shows: image, ground truth, prediction, and dice scores.

    Args:
        viz_data: List of dictionaries containing visualization data for each sample
        output_path: Path to save the PDF file
    """
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)

    for sample_idx, sample in enumerate(viz_data):
        image = sample['image'][0]  # Shape: [C, H, W, D]
        target = sample['target'][0]  # Shape: [C, H, W, D]
        prediction = sample['prediction'][0]  # Shape: [C, H, W, D]
        dice_scores = sample['dice_scores']  # Shape: [3]
        sample_name = sample['sample_name']

        # Select middle slice for visualization
        mid_slice = image.shape[-1] // 2

        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Sample: {sample_name}\nDice Scores - TC: {dice_scores[0]:.4f}, WT: {dice_scores[1]:.4f}, ET: {dice_scores[2]:.4f}, Avg: {dice_scores.mean():.4f}',
                     fontsize=14, fontweight='bold')

        # First row: Images (modalities)
        # Show FLAIR modality (channel 0)
        axes[0, 0].imshow(image[0, :, :, mid_slice], cmap='gray')
        axes[0, 0].set_title('Image (FLAIR)')
        axes[0, 0].axis('off')

        # Ground truth for each class
        axes[0, 1].imshow(target[0, :, :, mid_slice], cmap='Reds', alpha=0.7)
        axes[0, 1].set_title('Ground Truth - TC')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(target[1, :, :, mid_slice], cmap='Greens', alpha=0.7)
        axes[0, 2].set_title('Ground Truth - WT')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(target[2, :, :, mid_slice], cmap='Blues', alpha=0.7)
        axes[0, 3].set_title('Ground Truth - ET')
        axes[0, 3].axis('off')

        # Second row: Predictions
        axes[1, 0].imshow(image[0, :, :, mid_slice], cmap='gray')
        axes[1, 0].set_title('Image (FLAIR)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(prediction[0, :, :, mid_slice], cmap='Reds', alpha=0.7)
        axes[1, 1].set_title(f'Prediction - TC (Dice: {dice_scores[0]:.4f})')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(prediction[1, :, :, mid_slice], cmap='Greens', alpha=0.7)
        axes[1, 2].set_title(f'Prediction - WT (Dice: {dice_scores[1]:.4f})')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(prediction[2, :, :, mid_slice], cmap='Blues', alpha=0.7)
        axes[1, 3].set_title(f'Prediction - ET (Dice: {dice_scores[2]:.4f})')
        axes[1, 3].axis('off')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        print(f"Generated visualization page {sample_idx + 1}/{len(viz_data)}: {sample_name}")

    pdf.close()
    print(f"\nPDF saved to: {output_path}")


parser = argparse.ArgumentParser(description="TextBraTS segmentation pipeline")
parser.add_argument("--data_dir", default="./data/TextBraTSData", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="TextBraTS", type=str, help="experiment name")
parser.add_argument("--json_list", default="Test.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument(
    "--pretrained_dir",
    default="./runs/TextBraTS/",
    type=str,
    help="pretrained checkpoint directory",
)
parser.add_argument("--without_text", action="store_true", help="disable text features")
parser.add_argument("--use_v2", action="store_true", help="use TextSwinUNETR_V2 instead of TextSwinUNETR")
parser.add_argument("--fusion_decoder", default=None, type=str, choices=["jf", "tf"], help="text-fusion decoder type: jf (joint fusion), tf (text fusion), or none")
parser.add_argument("--spatial_prompting", action="store_true", help="use atlas masks as spatial prompts for the network")
parser.add_argument("--atlas_masks_dir", default="/Disk1/afrouz/Data/TextBraTS_atlas_masks", type=str, help="directory containing per-sample atlas masks")
parser.add_argument("--visualize", action="store_true", help="generate PDF visualization of test results")


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    test_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    # Since CUDA_VISIBLE_DEVICES=1 is set, physical GPU 1 appears as GPU 0
    gpu_index = 1
    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        print(f"GPU {gpu_index} not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
        print(f"Successfully set default device to GPU: {gpu_index}")
        
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    if args.fusion_decoder == "jf":
        model = TextSwinUNETRJFDecoder(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
            text_dim=768,
            # use_text=not args.without_text,
        )
        print(f"Using TextSwinUNETR with Joint Fusion (JF) Decoder!")
    elif args.fusion_decoder == "tf":
        model = TextSwinUNETRTFDecoder(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
            text_dim=768,
            # use_text=not args.without_text,
        )
        print(f"Using TextSwinUNETR with Text Fusion (TF) Decoder!")
    elif args.use_v2:
        model = TextSwinUNETR_V2(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
            text_dim=768,
            use_text=not args.without_text,
        )
        print(f"Using SwinUnetR Version 2!")
    else:
        model = TextSwinUNETR(
            img_size=128,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
            text_dim=768,
            use_text=not args.without_text,
        )
        print(f"Using TextSwinUNETR!")
    model_dict = torch.load(pretrained_pth, weights_only=False)["state_dict"]
    model.load_state_dict(model_dict, strict=False)
    model.eval()
    model.to(device)

    def val_epoch(model, loader, acc_func, hd95_func, visualize=False):
        model.eval()
        start_time = time.time()
        run_acc = AverageMeter()
        run_hd95 = AverageMeter()

        # Storage for visualization data
        viz_data = [] if visualize else None

        with torch.no_grad():
            for idx, batch_data in enumerate(loader):
                data = batch_data["image"]
                target = batch_data["label"]
                text = batch_data["text_feature"]
                atlas_mask = batch_data.get("atlas_mask", None)

                data, target, text = data.to(device), target.to(device), text.to(device)
                if atlas_mask is not None:
                    atlas_mask = atlas_mask.to(device)

                if atlas_mask is not None:
                    logits = model(data, text, atlas_mask=atlas_mask)
                else:
                    logits = model(data, text)
                prob = torch.sigmoid(logits)
                prob = (prob > 0.5).int()

                acc_func(y_pred=prob, y=target)
                acc, not_nans = acc_func.aggregate()
                acc = acc.to(device)

                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

                # HD95 Metric
                hd95_func(y_pred=prob, y=target)
                hd95 = hd95_func.aggregate()
                run_hd95.update(hd95.cpu().numpy())

                # Store data for visualization
                if visualize:
                    # Compute per-sample dice scores
                    sample_dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
                    sample_dice_metric(y_pred=prob, y=target)
                    sample_dice, _ = sample_dice_metric.aggregate()

                    viz_data.append({
                        'image': data.cpu().numpy(),
                        'target': target.cpu().numpy(),
                        'prediction': prob.cpu().numpy(),
                        'dice_scores': sample_dice.cpu().numpy(),
                        'sample_name': batch_data.get('name', [f'sample_{idx}'])[0]
                    })

                Dice_TC = run_acc.avg[0]
                Dice_WT = run_acc.avg[1]
                Dice_ET = run_acc.avg[2]
                HD95_TC = run_hd95.avg[0]
                HD95_WT = run_hd95.avg[1]
                HD95_ET = run_hd95.avg[2]
                print(
                    "Val  {}/{}".format(idx, len(loader)),
                    ", Dice_TC:", Dice_TC,
                    ", Dice_WT:", Dice_WT,
                    ", Dice_ET:", Dice_ET,
                    ", Avg Dice:", (Dice_ET + Dice_TC + Dice_WT) / 3,
                    ", HD95_TC:", HD95_TC,
                    ", HD95_WT:", HD95_WT,
                    ", HD95_ET:", HD95_ET,
                    ", Avg HD95:", (HD95_ET + HD95_TC + HD95_WT) / 3,
                    ", time {:.2f}s".format(time.time() - start_time),
                )
                start_time = time.time()
            with open(output_directory+'/log.txt', "a") as log_file:
                log_file.write(f"Experiment name:{args.pretrained_dir.split('/')[-2]}, "
            f"Final Validation Results - Dice_TC: {Dice_TC}, Dice_WT: {Dice_WT}, Dice_ET: {Dice_ET}, "
            f"Avg Dice: {(Dice_ET + Dice_TC + Dice_WT) / 3}, "
            f"HD95_TC: {HD95_TC}, HD95_WT: {HD95_WT}, HD95_ET: {HD95_ET}, "
            f"Avg HD95: {(HD95_ET + HD95_TC + HD95_WT) / 3}\n")
        return run_acc.avg, viz_data

    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    hd95_acc = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, percentile=95.0)
    avg_dice, viz_data = val_epoch(model, test_loader, acc_func=dice_acc, hd95_func=hd95_acc, visualize=args.visualize)

    # Generate PDF visualization if requested
    if args.visualize and viz_data:
        pdf_path = os.path.join(output_directory, 'test_visualizations.pdf')
        print(f"\nGenerating PDF visualization with {len(viz_data)} samples...")
        create_visualization_pdf(viz_data, pdf_path)

if __name__ == "__main__":
    main()
