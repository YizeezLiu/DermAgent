"""
Concept annotation dataset configurations for Task 2.
"""

from .base import DatasetConfig


DERM7PT_CONCEPTS = [
    "pigment_network",
    "blue_whitish_veil",
    "vascular_structures",
    "pigmentation",
    "streaks",
    "dots_and_globules",
    "regression_structures",
]

DERM7PT_PROMPT_TEMPLATES = [
    "a dermoscopy image with {} present",
    "a clinical photo showing {}",
    "skin lesion exhibiting {}",
]

DERM7PT_VQA_PROMPT = """You are a dermatology assistant performing concept annotation.
Look at the image and list ALL present dermoscopic concepts from:
pigment_network; blue_whitish_veil; vascular_structures; pigmentation; streaks; dots_and_globules; regression_structures.
Return a comma-separated list (e.g., pigment_network, streaks). If none apply, answer \"none\"."""

DERM7PT_CONCEPT_CONFIG = DatasetConfig(
    name="derm7pt_concepts",
    task="concepts",
    csv_path="datasets/derm7pt/meta_task2_sample_100.csv",
    data_root="datasets/derm7pt",
    class_names=DERM7PT_CONCEPTS,
    filename_col="ImageID",
    image_subdir="final_images",
    label_mode="multilabel",
    label_cols=DERM7PT_CONCEPTS,
    prompt_templates=DERM7PT_PROMPT_TEMPLATES,
    vqa_prompt=DERM7PT_VQA_PROMPT,
)


SKINCON_CONCEPTS = [
    "Vesicle",
    "Papule",
    "Macule",
    "Plaque",
    "Pustule",
    "Bulla",
    "Patch",
    "Nodule",
    "Ulcer",
    "Crust",
    "Erosion",
    "Excoriation",
    "Atrophy",
    "Exudate",
    "Fissure",
    "Induration",
    "Xerosis",
    "Telangiectasia",
    "Scale",
    "Scar",
    "Friable",
    "Pedunculated",
    "Exophytic/Fungating",
    "Warty/Papillomatous",
    "Dome-shaped",
    "Brown(Hyperpigmentation)",
    "White(Hypopigmentation)",
    "Purple",
    "Yellow",
    "Black",
    "Erythema",
    "Umbilicated",
]

SKINCON_PROMPT_TEMPLATES = [
    "a clinical skin image exhibiting {}",
    "dermatology photograph with {} finding",
    "skin lesion showing {}",
]

SKINCON_VQA_PROMPT = """You are a dermatology assistant identifying lesion morphology concepts.
From the provided list return ALL present findings as a comma-separated string, or \"none\" if absent:
Vesicle, Papule, Macule, Plaque, Pustule, Bulla, Patch, Nodule, Ulcer, Crust, Erosion, Excoriation,
Atrophy, Exudate, Fissure, Induration, Xerosis, Telangiectasia, Scale, Scar, Friable, Pedunculated,
Exophytic/Fungating, Warty/Papillomatous, Dome-shaped, Brown(Hyperpigmentation), White(Hypopigmentation),
Purple, Yellow, Black, Erythema, Umbilicated."""

SKINCON_CONCEPT_CONFIG = DatasetConfig(
    name="skincon_concepts",
    task="concepts",
    csv_path="datasets/skincon/meta_task2_sample_100.csv",
    data_root="datasets/skincon",
    class_names=SKINCON_CONCEPTS,
    filename_col="ImageID",
    image_subdir="final_images",
    label_mode="multilabel",
    label_cols=SKINCON_CONCEPTS,
    prompt_templates=SKINCON_PROMPT_TEMPLATES,
    vqa_prompt=SKINCON_VQA_PROMPT,
)


# =============================================================================
# 500-Sample Configs for Task 2
# =============================================================================

DERM7PT_500_CONCEPT_CONFIG = DatasetConfig(
    name="derm7pt_500",
    task="concepts",
    csv_path="datasets/derm7pt/meta_task2_sample_500.csv",
    data_root="datasets/derm7pt",
    class_names=DERM7PT_CONCEPTS,
    filename_col="ImageID",
    image_subdir="final_images",
    label_mode="multilabel",
    label_cols=DERM7PT_CONCEPTS,
    prompt_templates=DERM7PT_PROMPT_TEMPLATES,
    vqa_prompt=DERM7PT_VQA_PROMPT,
)

SKINCON_500_CONCEPT_CONFIG = DatasetConfig(
    name="skincon_500",
    task="concepts",
    csv_path="datasets/skincon/meta_task2_sample_500.csv",
    data_root="datasets/skincon",
    class_names=SKINCON_CONCEPTS,
    filename_col="ImageID",
    image_subdir="final_images",
    label_mode="multilabel",
    label_cols=SKINCON_CONCEPTS,
    prompt_templates=SKINCON_PROMPT_TEMPLATES,
    vqa_prompt=SKINCON_VQA_PROMPT,
)


__all__ = [
    "DERM7PT_CONCEPT_CONFIG",
    "SKINCON_CONCEPT_CONFIG",
    "DERM7PT_500_CONCEPT_CONFIG",
    "SKINCON_500_CONCEPT_CONFIG",
    "DERM7PT_CONCEPTS",
    "SKINCON_CONCEPTS",
]

