"""
Simple LangChain MRI Report Extractor for TextBraTS Dataset

Requirements:
    pip install langchain langchain-openai langchain-anthropic langchain-google-genai
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import os
import argparse
from pathlib import Path
import time

# =======================
# System Prompt with Instructions
# =======================

SYSTEM_PROMPT = """You are an expert Neuroradiologist and Data Scientist. Your goal is to extract structured anatomical and volumetric data from MRI reports for the TextBraTS dataset.

**Anatomical Regions:**
- Major Lobes: Frontal Lobe, Temporal Lobe, Parietal Lobe, Occipital Lobe
- Sub-regions: Insula, Basal Ganglia, Thalamus, Cerebellum, Brainstem
- Fluid Spaces: Lateral Ventricles, Third Ventricle, Fourth Ventricle

**Volumetric Descriptor Categories:**

**HIGH VOLUME/EXTENT:**
- "significant", "extensive", "marked", "diffuse", "prominent"
- "considerable", "pronounced", "massive", "large", "widespread"
- "substantial", "severe", "major", "gross", "abundant"

**LOW VOLUME/EXTENT:**
- "slight", "partial", "partially", "minor", "spotty"
- "speckled", "patchy", "small", "limited", "minimal"
- "mild", "trace", "scattered", "focal", "localized"

**MODERATE/UNCERTAIN:**
- "clear", "noticeable", "obvious", "certain degree of"
- "moderate", "intermediate", "present", "visible"

**SIGNAL/TEXTURE:**
- "heterogeneous", "homogeneous", "mixed", "irregular", "uniform"
- "high signal", "low signal", "enhanced", "speckled", "mottled"

**SPATIAL DISTRIBUTION:**
- "concentrated", "dispersed", "extends to", "involving"
- "centered", "peripheral", "multifocal", "confluent"

**MASS EFFECT:**
- "compressed", "deformed", "shifted", "displaced", "effaced"
- "pressure", "compression", "deviation", "obliterated"

**Rules:**
1. Identify the **Side** (Left/Right/Bilateral/Midline) and **Region**
2. For each pathology (Lesion, Edema, Necrosis, Mass_Effect), extract:
   - **Volumetric_Extent**: HIGH/LOW/MODERATE based on keywords above
   - **Descriptors**: All relevant keywords from the report
   - **Signal_Characteristics**: Signal intensity patterns (if applicable)
   - **Spatial_Pattern**: Distribution pattern (if applicable)
3. Extract ALL volumetric and qualitative descriptors mentioned
4. Output strictly valid JSON

**Output Format:**
{{
  "Report_ID": integer,
  "Pathologies": {{
    "Lesion": [
      {{
        "Region": "Anatomy",
        "Side": "Left/Right/Bilateral/Midline",
        "Volumetric_Extent": "HIGH/LOW/MODERATE",
        "Descriptors": ["keyword1", "keyword2", ...],
        "Signal_Characteristics": ["high signal", "low signal", "heterogeneous", ...],
        "Spatial_Pattern": ["concentrated", "dispersed", "extends to", ...]
      }}
    ],
    "Edema": [ ... same structure ... ],
    "Necrosis": [ ... same structure ... ],
    "Mass_Effect": [
      {{
        "Region": "Affected Structure",
        "Side": "Left/Right/Bilateral/Midline",
        "Volumetric_Extent": "HIGH/LOW/MODERATE",
        "Descriptors": ["compressed", "deformed", ...],
        "Effect_Type": ["compression", "displacement", "obliteration", ...]
      }}
    ]
  }},
  "Overall_Burden": {{
    "Lesion_Extent": "HIGH/LOW/MODERATE",
    "Edema_Extent": "HIGH/LOW/MODERATE",
    "Mass_Effect_Severity": "HIGH/LOW/MODERATE"
  }}
}}

**Example Extractions:**

Example Report 1:
"The brain MRI reveals a large lesion in the right frontal and temporal lobes, with the frontal lobe showing a mixture of high and low signal intensities, including some speckled high signals, while the temporal lobe displays more prominent high-signal areas accompanied by low signals. The right parietal lobe also shows a mix of high and low signals. Peritumoral edema is mainly concentrated in the right cerebral hemisphere, particularly noticeable in the frontal and temporal lobes, with significant swelling observed, indicating a large range of edema. Necrotic regions are present, with necrosis concentrated in the temporal lobe in one large region, although there is no obvious widespread necrosis. The lesion exerts pressure on the third ventricle and the right lateral ventricle, leading to deformation in both structures."

Example Output 1:
{{
  "Report_ID": 1,
  "Pathologies": {{
    "Lesion": [
      {{
        "Region": "Frontal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "HIGH",
        "Descriptors": ["large", "mixture", "speckled"],
        "Signal_Characteristics": ["mixture high/low signal", "speckled high signals"],
        "Spatial_Pattern": ["involving frontal lobe"]
      }},
      {{
        "Region": "Temporal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "HIGH",
        "Descriptors": ["prominent", "large"],
        "Signal_Characteristics": ["prominent high-signal areas", "accompanied by low signals"],
        "Spatial_Pattern": ["involving temporal lobe"]
      }},
      {{
        "Region": "Parietal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["mix"],
        "Signal_Characteristics": ["mix of high and low signals"],
        "Spatial_Pattern": ["present in parietal lobe"]
      }}
    ],
    "Edema": [
      {{
        "Region": "Cerebral Hemisphere",
        "Side": "Right",
        "Volumetric_Extent": "HIGH",
        "Descriptors": ["significant", "large range", "mainly concentrated"],
        "Signal_Characteristics": ["high signal"],
        "Spatial_Pattern": ["mainly concentrated", "particularly noticeable in frontal and temporal"]
      }},
      {{
        "Region": "Frontal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "HIGH",
        "Descriptors": ["significant swelling", "particularly noticeable"],
        "Signal_Characteristics": ["high signal"],
        "Spatial_Pattern": ["concentrated"]
      }},
      {{
        "Region": "Temporal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "HIGH",
        "Descriptors": ["significant swelling", "particularly noticeable"],
        "Signal_Characteristics": ["high signal"],
        "Spatial_Pattern": ["concentrated"]
      }}
    ],
    "Necrosis": [
      {{
        "Region": "Temporal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["one large region", "concentrated", "no obvious widespread"],
        "Signal_Characteristics": ["low signal"],
        "Spatial_Pattern": ["concentrated in temporal lobe"]
      }}
    ],
    "Mass_Effect": [
      {{
        "Region": "Third Ventricle",
        "Side": "Midline",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["exerted pressure", "deformation"],
        "Effect_Type": ["compression", "deformation"]
      }},
      {{
        "Region": "Lateral Ventricle",
        "Side": "Right",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["exerted pressure", "deformation"],
        "Effect_Type": ["compression", "deformation"]
      }}
    ]
  }},
  "Overall_Burden": {{
    "Lesion_Extent": "HIGH",
    "Edema_Extent": "HIGH",
    "Mass_Effect_Severity": "MODERATE"
  }}
}}

Example Report 2:
"MRI shows a right frontal lobe lesion with irregular high/low signal and significant enhancement at edges. Clear edema extends to the parietal and temporal lobes. Central necrosis with irregular zones is present. The right lateral ventricle is slightly compressed with a slight change in shape."

Example Output 2:
{{
  "Report_ID": 2,
  "Pathologies": {{
    "Lesion": [
      {{
        "Region": "Frontal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["significant enhancement"],
        "Signal_Characteristics": ["irregular high/low signal", "significant enhancement at edges"],
        "Spatial_Pattern": ["frontal lobe"]
      }}
    ],
    "Edema": [
      {{
        "Region": "Frontal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["clear"],
        "Signal_Characteristics": ["high signal"],
        "Spatial_Pattern": ["extends to parietal and temporal"]
      }},
      {{
        "Region": "Parietal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["extends to"],
        "Signal_Characteristics": ["high signal"],
        "Spatial_Pattern": ["extending"]
      }},
      {{
        "Region": "Temporal Lobe",
        "Side": "Right",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["extends to"],
        "Signal_Characteristics": ["high signal"],
        "Spatial_Pattern": ["extending"]
      }}
    ],
    "Necrosis": [
      {{
        "Region": "Lesion Center",
        "Side": "Right",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["central", "irregular zones"],
        "Signal_Characteristics": ["low signal", "irregular"],
        "Spatial_Pattern": ["concentrated in central part"]
      }}
    ],
    "Mass_Effect": [
      {{
        "Region": "Lateral Ventricle",
        "Side": "Right",
        "Volumetric_Extent": "LOW",
        "Descriptors": ["slightly compressed", "slight change in shape"],
        "Effect_Type": ["compression", "deformation"]
      }}
    ]
  }},
  "Overall_Burden": {{
    "Lesion_Extent": "MODERATE",
    "Edema_Extent": "MODERATE",
    "Mass_Effect_Severity": "LOW"
  }}
}}

Example Report 3:
"Left frontal and parietal lobe lesion with heterogeneous high and low signals. Clear high signal edema predominantly in frontal and parietal lobes. Necrosis is relatively dispersed with varied intensities. Bilateral ventricles are compressed with slight deformation and pressure effects."

Example Output 3:
{{
  "Report_ID": 3,
  "Pathologies": {{
    "Lesion": [
      {{
        "Region": "Frontal Lobe",
        "Side": "Left",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["heterogeneous"],
        "Signal_Characteristics": ["heterogeneous high and low signals"],
        "Spatial_Pattern": ["involving frontal and parietal"]
      }},
      {{
        "Region": "Parietal Lobe",
        "Side": "Left",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["heterogeneous"],
        "Signal_Characteristics": ["heterogeneous high and low signals"],
        "Spatial_Pattern": ["involving frontal and parietal"]
      }}
    ],
    "Edema": [
      {{
        "Region": "Frontal Lobe",
        "Side": "Left",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["clear", "predominantly"],
        "Signal_Characteristics": ["clear high signal"],
        "Spatial_Pattern": ["predominantly in frontal and parietal"]
      }},
      {{
        "Region": "Parietal Lobe",
        "Side": "Left",
        "Volumetric_Extent": "MODERATE",
        "Descriptors": ["clear", "predominantly"],
        "Signal_Characteristics": ["clear high signal"],
        "Spatial_Pattern": ["predominantly in frontal and parietal"]
      }}
    ],
    "Necrosis": [
      {{
        "Region": "Unspecified",
        "Side": "Left",
        "Volumetric_Extent": "LOW",
        "Descriptors": ["relatively dispersed", "varied intensities"],
        "Signal_Characteristics": ["varied intensities"],
        "Spatial_Pattern": ["relatively dispersed"]
      }}
    ],
    "Mass_Effect": [
      {{
        "Region": "Ventricles",
        "Side": "Bilateral",
        "Volumetric_Extent": "LOW",
        "Descriptors": ["compressed", "slight deformation", "pressure effects"],
        "Effect_Type": ["compression", "deformation", "pressure"]
      }}
    ]
  }},
  "Overall_Burden": {{
    "Lesion_Extent": "MODERATE",
    "Edema_Extent": "MODERATE",
    "Mass_Effect_Severity": "LOW"
  }}
}}

Now extract from the following report."""

# =======================
# Prompt Template
# =======================

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Report ID: {report_id}\n\nReport Text:\n{report_text}")
])

# =======================
# LLM Setup Functions
# =======================

def extract_with_openai(report_text, report_id, api_key):
    """Extract using OpenAI (GPT-4, GPT-4o, etc.)"""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o",
        temperature=0
    )

    chain = prompt_template | llm | JsonOutputParser()
    result = chain.invoke({"report_text": report_text, "report_id": report_id})
    return result


def extract_with_anthropic(report_text, report_id, api_key):
    """Extract using Anthropic Claude"""
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        api_key=api_key,
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )

    chain = prompt_template | llm | JsonOutputParser()
    result = chain.invoke({"report_text": report_text, "report_id": report_id})
    return result


def extract_with_gemini(report_text, report_id, api_key):
    """Extract using Google Gemini"""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key,
        temperature=0
    )

    chain = prompt_template | llm | JsonOutputParser()
    result = chain.invoke({"report_text": report_text, "report_id": report_id})
    return result


def extract_batch(reports, provider="openai", api_key=None):
    """
    Extract from multiple reports.

    Args:
        reports: List of dicts with 'id' and 'text' keys
        provider: "openai", "anthropic", or "gemini"
        api_key: API key for the provider

    Returns:
        List of extraction results
    """
    extract_func = {
        "openai": extract_with_openai,
        "anthropic": extract_with_anthropic,
        "gemini": extract_with_gemini
    }[provider]

    results = []
    for i, report in enumerate(reports, 1):
        print(f"Processing report {i}/{len(reports)}...")
        try:
            result = extract_func(report['text'], report['id'], api_key)
            results.append(result)
            print(f"✓ Report {report['id']} extracted successfully")
        except Exception as e:
            print(f"✗ Error on report {report['id']}: {e}")
            results.append({"Report_ID": report['id'], "error": str(e)})

    return results


def process_all_reports(data_dir: str, output_json: str, api_key: str, provider: str = "anthropic"):
    """
    Process all text reports in the data directory and save volumetric extractions.

    Args:
        data_dir: Path to directory containing BraTS samples
        output_json: Path to save the output JSON file
        api_key: API key for the LLM provider
        provider: LLM provider ("openai", "anthropic", or "gemini")
    """

    data_path = Path(data_dir)

    # Load existing results if file exists
    if Path(output_json).exists():
        print(f"Loading existing results from {output_json}")
        with open(output_json, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Found {len(results)} existing extractions")
    else:
        results = {}

    # Find all sample directories
    sample_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("BraTS")])

    print(f"Found {len(sample_dirs)} samples to process")

    # Count how many need processing
    to_process = [d for d in sample_dirs if d.name not in results or "error" in results.get(d.name, {})]
    print(f"{len(to_process)} samples need processing (skipping {len(sample_dirs) - len(to_process)} already completed)")

    for i, sample_dir in enumerate(sample_dirs):
        sample_id = sample_dir.name

        # Skip if already processed successfully
        if sample_id in results and "error" not in results[sample_id]:
            print(f"Skipping {i+1}/{len(sample_dirs)}: {sample_id} (already processed)")
            continue

        text_file = sample_dir / f"{sample_id}_flair_text.txt"

        if not text_file.exists():
            print(f"Warning: Text file not found for {sample_id}")
            continue

        print(f"Processing {i+1}/{len(sample_dirs)}: {sample_id}")

        with open(text_file, 'r') as f:
            text_report = f.read().strip()

        # Extract using the specified provider
        try:
            extract_func = {
                "openai": extract_with_openai,
                "anthropic": extract_with_anthropic,
                "gemini": extract_with_gemini
            }[provider]

            result = extract_func(text_report, sample_id, api_key)
            result["Sample_ID"] = sample_id  # Add sample ID to result
            result["Original_Text"] = text_report  # Store the original text report
            results[sample_id] = result
            print(f"✓ Successfully processed {sample_id}")
        except Exception as e:
            print(f"✗ Error processing {sample_id}: {e}")
            results[sample_id] = {
                "Sample_ID": sample_id,
                "Original_Text": text_report,
                "error": str(e)
            }

        # Save results after EVERY iteration
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  → Saved progress to {output_json} ({len(results)}/{len(sample_dirs)} completed)")

    print(f"\n✓ Final save: {len(results)} extractions to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Extract anatomical volumetric data from BraTS text reports")
    parser.add_argument("--data_dir", type=str, default="/Disk1/afrouz/Data/Merged",
                        help="Path to BraTS data directory")
    parser.add_argument("--output", type=str, default="volumetric_extractions.json",
                        help="Output JSON file path")
    parser.add_argument("--provider", type=str, default="anthropic",
                        choices=["openai", "anthropic", "gemini"],
                        help="LLM provider to use")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY/GOOGLE_API_KEY env var)")
    parser.add_argument("--sample", type=str, default=None,
                        help="Process only a single sample (e.g., BraTS20_Training_009)")

    args = parser.parse_args()

    # Get API key based on provider
    if args.api_key:
        api_key = args.api_key
    elif args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
    elif args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    elif args.provider == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY")
    else:
        api_key = None

    if not api_key:
        raise ValueError(f"Please provide API key via --api_key or set the appropriate environment variable for {args.provider.upper()}")

    if args.sample:
        # Process single sample for testing
        sample_dir = Path(args.data_dir) / args.sample
        text_file = sample_dir / f"{args.sample}_flair_text.txt"

        if not text_file.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")

        with open(text_file, 'r') as f:
            text_report = f.read().strip()

        print(f"Text report:\n{text_report}\n")
        print(f"Extracting with {args.provider.upper()}...")

        # Extract data
        extract_func = {
            "openai": extract_with_openai,
            "anthropic": extract_with_anthropic,
            "gemini": extract_with_gemini
        }[args.provider]

        result = extract_func(text_report, args.sample, api_key)

        print("\nExtracted data:")
        print(json.dumps(result, indent=2))

        # Save single sample result
        output_file = f"{args.sample}_extraction.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved to {output_file}")
    else:
        # Process all samples
        process_all_reports(args.data_dir, args.output, api_key, args.provider)


if __name__ == "__main__":
    main()
