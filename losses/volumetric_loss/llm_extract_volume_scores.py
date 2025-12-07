"""
LLM-based extraction of volume dominance scores from BraTS text reports.
Extracts descriptors to determine whether edema or tumor core is dominant.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple
import anthropic


# Volume descriptor categories
HIGH_VOLUME_DESCRIPTORS = {
    "significant", "extensive", "marked", "diffuse", "prominent",
    "considerable", "pronounced", "massive", "large", "widespread"
}

LOW_VOLUME_DESCRIPTORS = {
    "slight", "partial", "partially", "minor", "spotty",
    "speckled", "patchy", "small", "limited", "minimal"
}

NEUTRAL_DESCRIPTORS = {
    "clear", "noticeable", "obvious", "certain degree of"
}


def extract_volume_dominance_llm(text_report: str, api_key: str) -> Dict:
    """
    Use Claude to extract volume dominance information from text report.

    Returns:
        dict with keys:
            - edema_delta: dominance score for edema (-1, 0, +1)
            - core_delta: dominance score for tumor core (-1, 0, +1)
            - overall_delta: overall dominance (-1 = core dominant, 0 = balanced, +1 = edema dominant)
            - edema_descriptors: list of descriptors found for edema
            - core_descriptors: list of descriptors found for tumor core/lesion
            - reasoning: explanation from LLM
    """

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are analyzing a brain tumor MRI text report to extract volume-related information.

Text Report:
{text_report}

Your task is to analyze the volume/extent descriptors for:
1. EDEMA (swelling, surrounding areas)
2. TUMOR CORE/LESION (necrosis, enhancing tumor, main lesion)

Volume Descriptor Categories:
HIGH VOLUME (large, dominant): significant, extensive, marked, diffuse, prominent, considerable, pronounced, massive, widespread
LOW VOLUME (small, minor): slight, partial/partially, minor, spotty, speckled, patchy, small, limited, minimal
NEUTRAL: clear, noticeable, obvious

Instructions:
1. Identify all descriptors related to EDEMA volume/extent
2. Identify all descriptors related to TUMOR CORE/LESION volume/extent
3. Determine dominance scores:
   - Score +1 if HIGH VOLUME descriptors dominate
   - Score -1 if LOW VOLUME descriptors dominate
   - Score 0 if neutral, balanced, or unclear

4. Calculate OVERALL dominance:
   - +1 if edema is the dominant feature overall
   - -1 if tumor core is the dominant feature overall
   - 0 if balanced or unclear

Return ONLY a valid JSON object with this exact structure:
{{
    "edema_delta": <-1, 0, or +1>,
    "core_delta": <-1, 0, or +1>,
    "overall_delta": <-1, 0, or +1>,
    "edema_descriptors": [list of descriptors found],
    "core_descriptors": [list of descriptors found],
    "reasoning": "brief explanation"
}}"""

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text

    # Parse JSON response
    try:
        # Extract JSON from response (handle potential markdown formatting)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response: {response_text}")
        # Return default neutral scores
        return {
            "edema_delta": 0,
            "core_delta": 0,
            "overall_delta": 0,
            "edema_descriptors": [],
            "core_descriptors": [],
            "reasoning": "Error parsing response"
        }


def process_all_reports(data_dir: str, output_json: str, api_key: str):
    """
    Process all text reports in the data directory and save dominance scores.

    Args:
        data_dir: Path to directory containing BraTS samples
        output_json: Path to save the output JSON file
        api_key: Anthropic API key
    """

    data_path = Path(data_dir)
    results = {}

    # Find all sample directories
    sample_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("BraTS")])

    print(f"Found {len(sample_dirs)} samples to process")

    for i, sample_dir in enumerate(sample_dirs):
        sample_id = sample_dir.name
        text_file = sample_dir / f"{sample_id}_flair_text.txt"

        if not text_file.exists():
            print(f"Warning: Text file not found for {sample_id}")
            continue

        print(f"Processing {i+1}/{len(sample_dirs)}: {sample_id}")

        # Read text report
        with open(text_file, 'r') as f:
            text_report = f.read().strip()

        # Extract dominance scores using LLM
        scores = extract_volume_dominance_llm(text_report, api_key)

        # Store results
        results[sample_id] = {
            "text_report": text_report,
            "edema_delta": scores["edema_delta"],
            "core_delta": scores["core_delta"],
            "overall_delta": scores["overall_delta"],
            "edema_descriptors": scores["edema_descriptors"],
            "core_descriptors": scores["core_descriptors"],
            "reasoning": scores["reasoning"]
        }

        print(f"  Edema δ={scores['edema_delta']}, Core δ={scores['core_delta']}, Overall δ={scores['overall_delta']}")

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {output_json}")
    print(f"Processed {len(results)} samples")


def main():
    parser = argparse.ArgumentParser(description="Extract volume dominance scores from BraTS text reports")
    parser.add_argument("--data_dir", type=str, default="/Disk1/afrouz/Data/Merged",
                        help="Path to BraTS data directory")
    parser.add_argument("--output", type=str, default="volume_dominance_scores.json",
                        help="Output JSON file path")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--sample", type=str, default=None,
                        help="Process only a single sample (e.g., BraTS20_Training_009)")

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please provide API key via --api_key or ANTHROPIC_API_KEY environment variable")

    if args.sample:
        # Process single sample for testing
        sample_dir = Path(args.data_dir) / args.sample
        text_file = sample_dir / f"{args.sample}_flair_text.txt"

        if not text_file.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")

        with open(text_file, 'r') as f:
            text_report = f.read().strip()

        print(f"Text report:\n{text_report}\n")

        scores = extract_volume_dominance_llm(text_report, api_key)
        print("\nExtracted scores:")
        print(json.dumps(scores, indent=2))
    else:
        # Process all samples
        process_all_reports(args.data_dir, args.output, api_key)


if __name__ == "__main__":
    main()
