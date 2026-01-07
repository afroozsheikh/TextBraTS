#!/usr/bin/env python3
"""
Create JSON files with dual text features (original flair_text + enriched_text)
for late fusion training.
"""

import json
import os
from pathlib import Path


def create_dual_text_json(flair_json_path, enriched_json_path, output_json_path):
    """
    Merge two JSON files to create a new one with both text features.

    Args:
        flair_json_path: Path to JSON with flair_text.npy references
        enriched_json_path: Path to JSON with enriched_text.npy references
        output_json_path: Output path for merged JSON
    """
    # Load both JSON files
    with open(flair_json_path, 'r') as f:
        flair_data = json.load(f)

    with open(enriched_json_path, 'r') as f:
        enriched_data = json.load(f)

    # Create new structure
    dual_data = {"training": []}

    # Iterate through and merge
    for flair_item, enriched_item in zip(flair_data["training"], enriched_data["training"]):
        # Verify they're the same sample
        assert flair_item["fold"] == enriched_item["fold"], "Fold mismatch!"
        assert flair_item["image"] == enriched_item["image"], "Image paths mismatch!"
        assert flair_item["label"] == enriched_item["label"], "Label paths mismatch!"

        # Create merged item
        merged_item = {
            "fold": flair_item["fold"],
            "image": flair_item["image"],
            "label": flair_item["label"],
            "text_feature": flair_item["text_feature"],  # Original flair text
            "enriched_text_feature": enriched_item["text_feature"]  # Enriched text
        }

        dual_data["training"].append(merged_item)

    # Write output
    with open(output_json_path, 'w') as f:
        json.dump(dual_data, f, indent=4)

    print(f"Created {output_json_path} with {len(dual_data['training'])} samples")
    print(f"Each sample now has:")
    print(f"  - text_feature: {dual_data['training'][0]['text_feature']}")
    print(f"  - enriched_text_feature: {dual_data['training'][0]['enriched_text_feature']}")


if __name__ == "__main__":
    base_dir = "/Disk1/afrouz/Projects/TextBraTS"

    # Create dual text JSON for test set
    print("=" * 60)
    print("Creating Test JSON with dual text features...")
    print("=" * 60)
    create_dual_text_json(
        flair_json_path=os.path.join(base_dir, "Test.json"),
        enriched_json_path=os.path.join(base_dir, "Test_extended_text_features.json"),
        output_json_path=os.path.join(base_dir, "Test_dual_text_features.json")
    )

    print("\n" + "=" * 60)
    print("Creating Train JSON with dual text features...")
    print("=" * 60)
    # Create dual text JSON for train set
    create_dual_text_json(
        flair_json_path=os.path.join(base_dir, "Train.json"),
        enriched_json_path=os.path.join(base_dir, "Train_extended_text_features.json"),
        output_json_path=os.path.join(base_dir, "Train_dual_text_features.json")
    )

    print("\n" + "=" * 60)
    print("Done! Created both JSON files with dual text features.")
    print("=" * 60)
