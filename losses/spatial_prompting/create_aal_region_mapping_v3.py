import json
from nilearn import datasets

def generate_corrected_mapping():
    print("Fetching AAL Atlas metadata...")
    aal = datasets.fetch_atlas_aal()
    
    # 1. Create the Correct Lookup Dictionary
    # specific_ids are strings in the dataset ('2001', etc.), we convert to int
    valid_ids = [int(i) for i in aal.indices]
    labels = [str(l) for l in aal.labels]
    
    # Map Name -> Real NIfTI ID (e.g., "Precentral_L" -> 2001)
    # Note: aal.indices and aal.labels are aligned lists
    name_to_id = dict(zip(labels, valid_ids))
    
    print(f"Loaded {len(name_to_id)} labels.")
    print(f"Sample Check: 'Precentral_L' maps to ID {name_to_id.get('Precentral_L')}")

    # 2. Define the Region Logic (Semantic Mapping)
    # We look for keywords in the AAL names to group them into lobes
    
    definitions = {
        "Frontal Lobe": ["Frontal", "Precentral", "Rolandic_Oper", "Supp_Motor_Area", "Olfactory", "Rectus", "Paracentral"],
        "Temporal Lobe": ["Temporal", "Heschl", "Fusiform"],
        "Parietal Lobe": ["Parietal", "Postcentral", "SupraMarginal", "Angular", "Precuneus"],
        "Occipital Lobe": ["Occipital", "Calcarine", "Cuneus", "Lingual"],
        "Insula": ["Insula"],
        "Cingulate": ["Cingulum"], # AAL specific
        "Hippocampus": ["Hippocampus", "ParaHippocampal"],
        "Amygdala": ["Amygdala"],
        "Basal Ganglia": ["Caudate", "Putamen", "Pallidum"],
        "Thalamus": ["Thalamus"],
        "Cerebellum": ["Cerebelum", "Vermis"], # Note spelling 'Cerebelum' in some AAL versions
    }

    region_mapping = {}

    for region_name, keywords in definitions.items():
        # Find all names containing these keywords
        matches = []
        for name, real_id in name_to_id.items():
            if any(k in name for k in keywords):
                matches.append((name, real_id))
        
        # Split into Left, Right, Bilateral
        left_ids = [mid for name, mid in matches if "_L" in name or "Left" in name]
        right_ids = [mid for name, mid in matches if "_R" in name or "Right" in name]
        
        # Bilateral includes everything found
        bilateral_ids = [mid for name, mid in matches]

        region_mapping[region_name] = {
            "Left": sorted(left_ids),
            "Right": sorted(right_ids),
            "Bilateral": sorted(bilateral_ids),
            # "Unspecified" usually defaults to Bilateral for spatial prompting
            "Unspecified": sorted(bilateral_ids) 
        }

    # 3. Add Composite Regions (Useful for clinical prompting)
    
    # Frontal-Parietal Junction (often relevant for motor/sensory)
    fp_junction = region_mapping["Frontal Lobe"]["Bilateral"] + region_mapping["Parietal Lobe"]["Bilateral"]
    region_mapping["Junction of Frontal and Parietal Lobes"] = {
        "Left": region_mapping["Frontal Lobe"]["Left"] + region_mapping["Parietal Lobe"]["Left"],
        "Right": region_mapping["Frontal Lobe"]["Right"] + region_mapping["Parietal Lobe"]["Right"],
        "Bilateral": sorted(list(set(fp_junction))),
        "Unspecified": sorted(list(set(fp_junction)))
    }

    # Mesial Temporal (Hippocampus + Amygdala + Parahippocampal)
    mtl_bilateral = region_mapping["Hippocampus"]["Bilateral"] + region_mapping["Amygdala"]["Bilateral"]
    region_mapping["Mesial Temporal Lobe"] = {
        "Left": region_mapping["Hippocampus"]["Left"] + region_mapping["Amygdala"]["Left"],
        "Right": region_mapping["Hippocampus"]["Right"] + region_mapping["Amygdala"]["Right"],
        "Bilateral": sorted(list(set(mtl_bilateral))),
        "Unspecified": sorted(list(set(mtl_bilateral)))
    }

    # Cerebral Hemisphere (Frontal + Temporal + Parietal + Occipital)
    # Excluding Cerebellum and Deep Grey for strict "Hemisphere" definitions often used in radio
    hemi_bilateral = (region_mapping["Frontal Lobe"]["Bilateral"] + 
                      region_mapping["Temporal Lobe"]["Bilateral"] + 
                      region_mapping["Parietal Lobe"]["Bilateral"] + 
                      region_mapping["Occipital Lobe"]["Bilateral"])
    
    region_mapping["Cerebral Hemisphere"] = {
         "Left": (region_mapping["Frontal Lobe"]["Left"] + 
                  region_mapping["Temporal Lobe"]["Left"] + 
                  region_mapping["Parietal Lobe"]["Left"] + 
                  region_mapping["Occipital Lobe"]["Left"]),
         "Right": (region_mapping["Frontal Lobe"]["Right"] + 
                   region_mapping["Temporal Lobe"]["Right"] + 
                   region_mapping["Parietal Lobe"]["Right"] + 
                   region_mapping["Occipital Lobe"]["Right"]),
         "Bilateral": sorted(list(set(hemi_bilateral))),
         "Unspecified": sorted(list(set(hemi_bilateral)))
    }

    # Whole Brain
    all_ids = sorted(list(valid_ids))
    region_mapping["Brain"] = {
        "Left": all_ids, # Approximation
        "Right": all_ids, # Approximation
        "Bilateral": all_ids,
        "Unspecified": all_ids
    }

    return region_mapping

if __name__ == "__main__":
    mapping = generate_corrected_mapping()
    
    output_path = "region_mapping_aal_v2.json"
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
        
    print(f"\n✓ Saved corrected mapping to {output_path}")
    
    # Verification Print
    print("\n--- Verification: Frontal Lobe Left ---")
    print(f"IDs: {mapping['Frontal Lobe']['Left']}")
    # 2001 is Precentral_L, 2101 is Frontal_Sup_L, etc.
    if 2001 in mapping['Frontal Lobe']['Left']:
        print("SUCCESS: ID 2001 (Precentral_L) is correctly mapped to Frontal Lobe Left.")
    else:
        print("FAILURE: ID 2001 is missing.")