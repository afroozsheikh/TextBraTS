# Pathology Definitions and Mapping to BraTS Labels

This document provides a precise mapping between pathology types extracted from radiology reports (in `volumetric_extractions.json`) and the standard BraTS segmentation labels.

---

## 1. Pathology Types in volumetric_extractions.json

Your [volumetric_extractions.json](volumetric_extractions.json) contains **4 pathology types** extracted from radiology reports:

### **Lesion**
- **Medical Definition**: Any abnormal tissue; in brain tumors, this generally refers to the entire tumor mass
- **Imaging Characteristics**: Mixed signal patterns (high/low intensities), representing the whole tumor area
- **Location**: Found in specific brain regions (lobes, hemispheres)

### **Edema**
- **Medical Definition**: Peritumoral edema - fluid accumulation and swelling in tissue surrounding the tumor
- **Imaging Characteristics**: High signal on T2/FLAIR sequences, appears as bright regions around the tumor
- **Clinical Significance**: Indicates tumor's effect on surrounding brain tissue, not part of the solid tumor itself

### **Necrosis**
- **Medical Definition**: Dead/dying tissue within the tumor core
- **Imaging Characteristics**: Low signal intensity, appears dark on contrast-enhanced images (doesn't enhance)
- **Clinical Significance**: Common in aggressive, fast-growing tumors like glioblastomas

### **Mass_Effect**
- **Medical Definition**: Compression, displacement, or distortion of surrounding brain structures due to tumor size
- **Examples**: Midline shift, ventricle compression, herniation
- **Important**: This is NOT a tissue type - it's a structural/spatial effect

---

## 2. BraTS Label Standard

The BraTS challenge uses these ground truth labels:

```
Label 0: Background (healthy brain)
Label 1: NCR/NET (Necrotic and Non-Enhancing Tumor Core)
Label 2: ED (Peritumoral Edema)
Label 4: ET (Enhancing Tumor)
```

These combine into **3 evaluation regions**:

- **Tumor Core (TC)**: Labels 1 + 4 (NCR/NET + ET)
- **Whole Tumor (WT)**: Labels 1 + 2 + 4 (all tumor tissue + edema)
- **Enhancing Tumor (ET)**: Label 4 only

---

## 3. Model Output Channels

Your model outputs 3 channels:

```
Channel 0 → TC (Tumor Core)
Channel 1 → WT (Whole Tumor)
Channel 2 → ET (Enhancing Tumor)
```

---

## 4. PRECISE MAPPING: JSON Pathologies ↔ BraTS Labels

```
┌─────────────────────────────────────────────────────────────────────┐
│              PATHOLOGY → BraTS LABEL MAPPING                        │
└─────────────────────────────────────────────────────────────────────┘

JSON Pathology  │ BraTS Component        │ Model Channel │ Calculation
────────────────┼────────────────────────┼───────────────┼──────────────
Lesion          │ Whole Tumor (WT)       │ Channel 1     │ Direct
                │ (entire tumor mass)    │               │
────────────────┼────────────────────────┼───────────────┼──────────────
Edema           │ Peritumoral Edema (ED) │ Part of WT    │ WT - TC
                │ (Label 2)              │               │ (Ch1 - Ch0)
────────────────┼────────────────────────┼───────────────┼──────────────
Necrosis        │ NCR/NET (Label 1)      │ Part of TC    │ TC - ET
                │ Non-enhancing core     │               │ (Ch0 - Ch2)
────────────────┼────────────────────────┼───────────────┼──────────────
Mass_Effect     │ Not a tissue label     │ N/A           │ N/A
                │ (structural effect)    │               │
────────────────┼────────────────────────┼───────────────┼──────────────
(Not in JSON)   │ Enhancing Tumor (ET)   │ Channel 2     │ Direct
                │ (Label 4)              │               │
```

---

## 5. Key Relationships - Visual Hierarchy

```
       ┌───────────────────────────────────────────┐
       │      Whole Tumor (WT) - Channel 1         │
       │  ┌──────────────────────────────────────┐ │
       │  │                                       │ │
       │  │    Edema (surrounds tumor)           │ │
       │  │                                       │ │
       │  └──────────────────────────────────────┘ │
       │         ┌──────────────────────┐          │
       │         │ Tumor Core (TC)      │          │
       │         │   Channel 0          │          │
       │         │  ┌────────────────┐  │          │
       │         │  │ Enhancing (ET) │  │          │
       │         │  │   Channel 2    │  │          │
       │         │  └────────────────┘  │          │
       │         │  ┌────────────────┐  │          │
       │         │  │ Necrosis       │  │          │
       │         │  │ (NCR/NET)      │  │          │
       │         │  └────────────────┘  │          │
       │         └──────────────────────┘          │
       └───────────────────────────────────────────┘
```

### Mathematical Relationships:
- **WT** = Edema + TC = Edema + Necrosis + ET
- **TC** = Necrosis + ET
- **Edema** = WT - TC
- **Necrosis** = TC - ET

---

## 6. Critical Notes for Implementation

1. **Lesion ≈ Whole Tumor (WT)**: In radiology reports, "lesion" is a catch-all term for the entire abnormal area, which maps most closely to WT (Channel 1)

2. **Edema is NOT the tumor**: It's swelling around the tumor, included in WT but NOT in TC

3. **Necrosis is inside TC**: Dead tissue within the tumor core, doesn't enhance on contrast imaging

4. **Mass Effect is not segmentable**: It describes structural changes (compression, shift) rather than a tissue type you can segment

5. **Your JSON doesn't explicitly mention "Enhancing Tumor"**: This is because radiology reports may describe enhancement patterns within "Lesion" descriptions rather than as a separate category

---

## 7. Example from Actual Data

Sample: **BraTS20_Training_001**

```json
"Lesion": [
  {"Region": "Frontal Lobe", "Side": "Right", "Volumetric_Extent": "MODERATE"}
]
```
→ Maps to **WT (Channel 1)**

```json
"Edema": [
  {"Region": "Parietal Lobe", "Side": "Right", "Volumetric_Extent": "HIGH"}
]
```
→ Part of **WT**, calculate as **WT - TC**

```json
"Necrosis": [
  {"Region": "Frontal Lobe", "Side": "Right", "Volumetric_Extent": "MODERATE"}
]
```
→ Part of **TC (Channel 0)**, calculate as **TC - ET**

```json
"Mass_Effect": [
  {"Region": "Lateral Ventricles", "Side": "Bilateral", "Volumetric_Extent": "HIGH"}
]
```
→ **Not mapped to segmentation** (describes compression effect)

---

## 8. Summary Table

| Pathology Type | Medical Meaning | BraTS Label | Model Channel | How to Extract |
|---|---|---|---|---|
| **Lesion** | Entire tumor mass | Whole Tumor (WT) | Channel 1 | Direct output |
| **Edema** | Swelling around tumor | Peritumoral Edema (ED) | Part of Channel 1 | WT - TC (Ch1 - Ch0) |
| **Necrosis** | Dead tissue in core | NCR/NET | Part of Channel 0 | TC - ET (Ch0 - Ch2) |
| **Mass_Effect** | Structural displacement | N/A | N/A | Not segmentable |
| **Enhancing Tumor** | Active tumor growth | ET | Channel 2 | Direct output |

---

## References

- BraTS 2020 Challenge: https://www.med.upenn.edu/cbica/brats2020/
- Standard BraTS evaluation metrics use TC, WT, and ET regions
- This mapping is based on standard radiological terminology and BraTS conventions