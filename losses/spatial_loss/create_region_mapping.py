"""
Create Region Mapping JSON for Spatial Loss

This script generates a comprehensive mapping from text-based anatomical region
descriptions to Harvard-Oxford atlas label IDs.

Medical Validation:
- Strictly separates Cortical Lobes from Deep White Matter and Subcortical Nuclei.
- Addresses the Bilateral nature of Harvard-Oxford Cortical labels (1-48).
- Corrects the mapping of Cingulate and Insular regions.

Author: TextBraTS Spatial Loss Implementation (Refactored)
Date: 2025-12-07
"""

import json
from typing import Dict, List

class RegionMappingGenerator:
    """Generate region mapping from text descriptions to atlas label IDs."""

    def __init__(self, atlas_labels_path: str):
        with open(atlas_labels_path, 'r') as f:
            self.atlas_labels = json.load(f)
        
        self._define_anatomical_groups()

    def _define_anatomical_groups(self):
        """
        Define anatomical groups based on Harvard-Oxford indices.
        IDs 1-48 are Cortical (Bilateral in this atlas).
        IDs 49-69 are Lateralized Subcortical/Volumetric.
        """

        # --- CORTICAL LOBES (Bilateral IDs 1-48) ---

        self.FRONTAL_LOBE = [
            1,  # Frontal Pole
            3,  # Superior Frontal Gyrus
            4,  # Middle Frontal Gyrus
            5,  # Inferior Frontal Gyrus, pars triangularis
            6,  # Inferior Frontal Gyrus, pars opercularis
            7,  # Precentral Gyrus
            25, # Frontal Medial Cortex
            26, # Juxtapositional Lobule (SMA)
            27, # Subcallosal Cortex
            28, # Paracingulate Gyrus
            33, # Frontal Orbital Cortex
            41, # Frontal Opercular Cortex
            # Note: Anterior Cingulate (29) is often grouped here clinically, 
            # but we keep it separate for precision unless requested.
        ]

        self.PARIETAL_LOBE = [
            17, # Postcentral Gyrus
            18, # Superior Parietal Lobule
            19, # Supramarginal Gyrus, anterior
            20, # Supramarginal Gyrus, posterior
            21, # Angular Gyrus
            31, # Precuneous Cortex
            43, # Parietal Opercular Cortex
        ]

        self.TEMPORAL_LOBE = [
            8,  # Temporal Pole
            9,  # Superior Temporal Gyrus, anterior
            10, # Superior Temporal Gyrus, posterior
            11, # Middle Temporal Gyrus, anterior
            12, # Middle Temporal Gyrus, posterior
            13, # Middle Temporal Gyrus, temporooccipital
            14, # Inferior Temporal Gyrus, anterior
            15, # Inferior Temporal Gyrus, posterior
            16, # Inferior Temporal Gyrus, temporooccipital
            34, # Parahippocampal Gyrus, anterior (Mesial Temporal)
            35, # Parahippocampal Gyrus, posterior (Mesial Temporal)
            37, # Temporal Fusiform Cortex, anterior
            38, # Temporal Fusiform Cortex, posterior
            39, # Temporal Occipital Fusiform Cortex
            44, # Planum Polare
            45, # Heschl's Gyrus
            46, # Planum Temporale
        ]

        self.OCCIPITAL_LOBE = [
            22, # Lateral Occipital Cortex, superior
            23, # Lateral Occipital Cortex, inferior
            24, # Intracalcarine Cortex
            32, # Cuneal Cortex
            36, # Lingual Gyrus
            40, # Occipital Fusiform Gyrus
            47, # Supracalcarine Cortex
            48, # Occipital Pole
        ]

        self.INSULA = [2] # Insular Cortex
        
        self.CINGULATE = [
            29, # Anterior Cingulate
            30, # Posterior Cingulate
        ]

        # --- SUBCORTICAL GREY MATTER ---
        # Strictly Nuclei (Excluding White Matter and generic "Cortex" masks)

        self.BASAL_GANGLIA_LEFT = [53, 54, 55, 59] # Caudate, Putamen, Pallidum, Accumbens
        self.BASAL_GANGLIA_RIGHT = [64, 65, 66, 69]

        self.THALAMUS_LEFT = [52]
        self.THALAMUS_RIGHT = [63]

        self.HIPPOCAMPUS_LEFT = [57]
        self.HIPPOCAMPUS_RIGHT = [67]

        self.AMYGDALA_LEFT = [58]
        self.AMYGDALA_RIGHT = [68]

        self.BRAINSTEM = [56]

        # --- VOLUMETRIC / GENERIC MASKS ---
        
        self.WM_LEFT = [49] # "Left Cerebral White Matter" (Coarse mask)
        self.WM_RIGHT = [60] # "Right Cerebral White Matter"
        
        # Generic "Cerebral Cortex" masks (Coarse)
        self.CORTEX_MASK_LEFT = [50] 
        self.CORTEX_MASK_RIGHT = [61]
        
        self.VENTRICLE_LEFT = [51]
        self.VENTRICLE_RIGHT = [62]

    def _normalize_region_name(self, region: str) -> str:
        region = region.lower().strip()
        region = region.replace("lobes", "lobe").replace("regions", "region").replace("areas", "area")
        return region

    def map_region_to_labels(self, region: str, side: str = "Unspecified") -> List[int]:
        """
        Map a region description to IDs.
        
        CRITICAL NOTE ON LATERALITY:
        Atlas labels 1-48 are Bilateral. If the input is "Left Frontal Lobe", 
        this function returns the IDs [1, 3, 4...]. It relies on the training loop 
        to apply a geometric left-hemisphere mask if strict laterality is required 
        for cortical regions.
        """
        normalized = self._normalize_region_name(region)
        labels = []

        # --- 1. LOBAR MAPPING ---
        # Note: We do NOT add white matter or subcortical nuclei to Lobe requests
        # to ensure the Spatial Loss focuses on the specific anatomical definition.

        if "frontal" in normalized:
            labels.extend(self.FRONTAL_LOBE)
            # Optional: Add Anterior Cingulate to Frontal requests if vague
            if "region" in normalized or "lobe" in normalized:
                labels.append(29) 

        if "parietal" in normalized:
            labels.extend(self.PARIETAL_LOBE)
            # Optional: Add Posterior Cingulate to Parietal
            if "region" in normalized or "lobe" in normalized:
                labels.append(30)

        if "temporal" in normalized:
            labels.extend(self.TEMPORAL_LOBE)

        if "occipital" in normalized:
            labels.extend(self.OCCIPITAL_LOBE)

        if "insula" in normalized:
            labels.extend(self.INSULA)

        # --- 2. SUBCORTICAL MAPPING ---
        
        # Basal Ganglia
        if "basal ganglia" in normalized or "striatum" in normalized:
            if side in ["Left", "Bilateral", "Unspecified"]:
                labels.extend(self.BASAL_GANGLIA_LEFT)
            if side in ["Right", "Bilateral", "Unspecified"]:
                labels.extend(self.BASAL_GANGLIA_RIGHT)
        
        # Thalamus
        if "thalamus" in normalized:
            if side in ["Left", "Bilateral", "Unspecified"]:
                labels.extend(self.THALAMUS_LEFT)
            if side in ["Right", "Bilateral", "Unspecified"]:
                labels.extend(self.THALAMUS_RIGHT)

        # Limbic (Hippocampus/Amygdala)
        if "hippocampus" in normalized or "mesial temporal" in normalized:
             if side in ["Left", "Bilateral", "Unspecified"]:
                labels.extend(self.HIPPOCAMPUS_LEFT)
             if side in ["Right", "Bilateral", "Unspecified"]:
                labels.extend(self.HIPPOCAMPUS_RIGHT)

        if "brainstem" in normalized:
            labels.extend(self.BRAINSTEM)

        # --- 3. GENERIC / WHOLE HEMISPHERE TERMS ---
        
        generic_terms = ["brain", "cerebrum", "hemisphere", "whole brain"]
        
        # If the term is vague (e.g. "Left Hemisphere"), we return the coarse masks
        # PLUS all specific structures to ensure full coverage.
        if any(term == normalized for term in generic_terms) or normalized == "brain tissue":
            
            # Left Hemisphere
            if side == "Left":
                # Add coarse masks + all left subcortical
                all_left = list(range(49, 60)) # 49-59 includes WM, Cortex, Nuclei
                # Add all cortical labels (1-48) because we can't separate them by ID alone
                # The geometric mask in training must handle the split.
                return sorted(list(set(all_left + list(range(1, 49)))))

            # Right Hemisphere
            elif side == "Right":
                all_right = list(range(60, 70))
                return sorted(list(set(all_right + list(range(1, 49)))))
            
            # Whole Brain
            else:
                return list(range(1, 70))

        # --- 4. FALLBACK / INTERSECTION HANDLING ---
        
        # If "Junction" is detected (e.g., "Frontal-Parietal Junction"), 
        # the logic above will have added BOTH Frontal and Parietal lists.
        # This is acceptable for spatial loss (union of regions).
        
        # Clean up
        return sorted(list(set(labels)))

    def generate_region_mapping(self) -> Dict:
        # Same generation logic as before, but using the stricter map_region_to_labels
        all_regions = [
            "Frontal Lobe", "Parietal Lobe", "Temporal Lobe", "Occipital Lobe",
            "Basal Ganglia", "Thalamus", "Brainstem", "Insula",
            "Cerebral Hemisphere", "Brain",
            "Junction of Frontal and Parietal Lobes",
            "Temporo-Parietal Region",
            "Mesial Temporal Lobe"
        ]
        sides = ["Left", "Right", "Bilateral", "Unspecified"]
        
        region_mapping = {}
        for region in all_regions:
            region_mapping[region] = {}
            for side in sides:
                region_mapping[region][side] = self.map_region_to_labels(region, side)
                
        return region_mapping

def main():
    # Update paths as needed
    atlas_labels_path = "/Disk1/afrouz/Data/TextBraTS_atlas_preprocess/atlas_labels_harvard-oxford.json"
    output_path = "/Disk1/afrouz/Projects/TextBraTS/losses/spatial_loss/region_mapping.json"
    
    generator = RegionMappingGenerator(atlas_labels_path)
    mapping = generator.generate_region_mapping()
    
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Mapping saved to {output_path}")

if __name__ == "__main__":
    main()