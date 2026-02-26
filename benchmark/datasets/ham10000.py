"""
HAM10000 Dataset Configuration.

HAM10000 is a large collection of multi-source dermatoscopic images of 
pigmented lesions for skin cancer classification.

Classes:
    - nv: melanocytic nevi (nevus)
    - mel: melanoma
    - bcc: basal cell carcinoma
    - akiec: actinic keratoses and intraepithelial carcinoma
    - bkl: benign keratosis-like lesions
    - df: dermatofibroma
    - vasc: vascular lesions
"""

from .base import DatasetConfig

# Class labels (short names in CSV -> full names)
HAM10000_CLASSES = [
    "melanocytic nevi",                    # nv (0)
    "melanoma",                            # mel (1)
    "basal cell carcinoma",                # bcc (2)
    "actinic keratosis",                   # akiec (3)
    "benign keratosis",                    # bkl (4)
    "dermatofibroma",                      # df (5)
    "vascular lesion",                     # vasc (6)
]

# Mapping from CSV dx values to class index
HAM10000_DX_TO_LABEL = {
    "nv": 0,
    "mel": 1,
    "bcc": 2,
    "akiec": 3,
    "bkl": 4,
    "df": 5,
    "vasc": 6,
}

# Prompt templates for zero-shot classification
HAM10000_PROMPT_TEMPLATES = [
    "a dermoscopy image of {}",
    "a clinical photo of {}",
    "a skin lesion of {}",
    "this is a photo of {}",
    "a dermatological image showing {}",
]

# VQA classification prompt
HAM10000_VQA_PROMPT = """Look at this skin lesion image carefully.
What is the most likely diagnosis? Choose exactly one from the following options:
1. melanocytic nevi
2. melanoma
3. basal cell carcinoma
4. actinic keratosis
5. benign keratosis
6. dermatofibroma
7. vascular lesion

Answer with only the diagnosis name, nothing else."""


HAM10000_CONFIG = DatasetConfig(
    name="HAM10000",
    task="diagnosis",
    csv_path="datasets/ham10000/HAM10000_benchmark_100.csv",
    data_root="datasets/ham10000",
    class_names=HAM10000_CLASSES,
    filename_col="image_id",  # Use image_id column
    label_col="dx",           # Use dx column
    image_ext=".jpg",         # Images are {image_id}.jpg
    label_map=HAM10000_DX_TO_LABEL,  # Map dx strings to label indices
    prompt_templates=HAM10000_PROMPT_TEMPLATES,
    vqa_prompt=HAM10000_VQA_PROMPT,
)

HAM10000_500_CONFIG = DatasetConfig(
    name="HAM10000_500",
    task="diagnosis",
    csv_path="datasets/ham10000/HAM10000_benchmark_500.csv",
    data_root="datasets/ham10000",
    class_names=HAM10000_CLASSES,
    filename_col="image_id",  # Use image_id column
    label_col="dx",           # Use dx column
    image_ext=".jpg",         # Images are {image_id}.jpg
    label_map=HAM10000_DX_TO_LABEL,  # Map dx strings to label indices
    prompt_templates=HAM10000_PROMPT_TEMPLATES,
    vqa_prompt=HAM10000_VQA_PROMPT,
)

