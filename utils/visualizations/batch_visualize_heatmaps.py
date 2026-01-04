"""
Batch visualization script to generate attention heatmaps for multiple samples.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def batch_visualize(
    model_path,
    data_dir,
    json_path,
    output_dir,
    num_samples=10,
    reduction_method='attention',
    multi_slice=False,
    n_slices=5
):
    """
    Generate visualizations for multiple samples.
    """
    print("="*80)
    print(f"Batch Visualization: Processing {num_samples} samples")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Path to visualization script
    script_path = os.path.join(
        os.path.dirname(__file__),
        'visualize_attention_heatmaps.py'
    )

    for sample_idx in range(num_samples):
        print(f"\n{'='*80}")
        print(f"Processing sample {sample_idx + 1}/{num_samples}")
        print(f"{'='*80}")

        # Build command
        cmd = [
            'python', script_path,
            '--model_path', model_path,
            '--data_dir', data_dir,
            '--json_path', json_path,
            '--output_dir', output_dir,
            '--sample_idx', str(sample_idx),
            '--reduction_method', reduction_method,
        ]

        if multi_slice:
            cmd.extend(['--multi_slice', '--n_slices', str(n_slices)])

        # Run visualization
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"✓ Successfully processed sample {sample_idx}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to process sample {sample_idx}: {e}")
            continue

    print(f"\n{'='*80}")
    print("Batch Visualization Complete!")
    print(f"{'='*80}")
    print(f"All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Batch visualize attention heatmaps')
    parser.add_argument('--model_path', type=str,
                       default='/Disk1/afrouz/Projects/TextBraTS/runs/TextBraTS_conda/model.pt',
                       help='Path to pretrained model')
    parser.add_argument('--data_dir', type=str,
                       default='/Disk1/afrouz/Data/Merged/',
                       help='Path to dataset directory')
    parser.add_argument('--json_path', type=str,
                       default='./Train.json',
                       help='Path to dataset JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='./utils/visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--reduction_method', type=str, default='attention',
                       choices=['mean', 'max', 'attention'],
                       help='Method to reduce channels to heatmap')
    parser.add_argument('--multi_slice', action='store_true',
                       help='Generate multi-slice visualization')
    parser.add_argument('--n_slices', type=int, default=5,
                       help='Number of slices for multi-slice visualization')

    args = parser.parse_args()

    batch_visualize(
        model_path=args.model_path,
        data_dir=args.data_dir,
        json_path=args.json_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        reduction_method=args.reduction_method,
        multi_slice=args.multi_slice,
        n_slices=args.n_slices
    )


if __name__ == '__main__':
    main()
