"""
Dataset configurations for DermAgent benchmark tasks.

This module is separated to avoid circular imports between benchmark_agent.py and tools.
"""

from typing import Dict, Any, List

# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Diagnosis
    "HAM10000": {
        "task": "diagnosis", 
        "options": ["melanocytic nevi", "melanoma", "basal cell carcinoma",
                   "actinic keratosis", "benign keratosis", "dermatofibroma", "vascular lesion"]
    },
    "SNU_500": {
        "task": "diagnosis",
        "options": [
            "abnormal", "abscess", "acanthosis nigricans", "acne", "actinic keratosis",
            "acute generalized exanthematous pustulosis", "alopecia areata", "amyloidosis",
            "androgenic alopecia", "angiofibroma", "angiokeratoma", "angular cheilitis",
            "atopic dermatitis", "basal cell carcinoma", "becker nevus", "blue nevus",
            "bowen disease", "bullous disease", "cafe au lait macule", "cellulitis",
            "condyloma", "confluent and reticulated papillomatosis", "congenital nevus",
            "contact dermatitis", "dermatofibroma", "drug eruption", "eczema herpeticum",
            "epidermal cyst", "epidermal nevus", "erythema ab igne", "erythema annulare centrifugum",
            "erythema multiforme", "erythema nodosum", "exfoliative dermatitis", "fifth disease",
            "folliculitis", "freckle", "furuncle", "granuloma annulare", "guttate psoriasis",
            "hand eczema", "hemangioma", "herpes simplex", "herpes zoster", "hypertrophic scar",
            "idiopathic guttate hypomelanosis", "impetigo", "inflammed cyst", "ingrowing nail",
            "insect bite", "juvenile xanthogranuloma", "keloid", "keratoacanthoma", "keratoderma",
            "keratosis pilaris", "lentigo", "lichen amyloidosis", "lichen nitidus", "lichen planus",
            "lichen simplex chronicus", "lichen striatus", "livedo reticularis", "lupus erythematosus",
            "lymphangioma", "malignant melanoma", "melanocytic nevus", "melanonychia", "melasma",
            "milia", "molluscum contagiosum", "morphea", "mucocele", "mucosal melanotic macule",
            "mucous cyst", "nail dystrophy", "neurofibroma", "neurofibromatosis", "nevus depigmentosus",
            "nevus spilus", "nummular eczema", "onycholysis", "onychomycosis", "organoid nevus",
            "ota nevus", "palmoplantar pustulosis", "paronychia", "perioral dermatitis",
            "pigmented progressive purpuric dermatosis", "pityriasis alba", "pityriasis lichenoides chronica",
            "pityriasis lichenoides et varioliformis acuta", "pityriasis rosea", "poikiloderma",
            "pompholyx", "porokeratosis", "poroma", "port wine stain", "prurigo nodularis",
            "prurigo pigmentosa", "psoriasis", "pustular psoriasis", "pyoderma gangrenosum",
            "pyogenic granuloma", "riehl melanosis", "rosacea", "scabies", "sebaceous hyperplasia",
            "seborrheic dermatitis", "seborrheic keratosis", "skin tag", "squamous cell carcinoma",
            "staphylococcal scalded skin syndrome", "steatocystoma multiplex", "striae distensae",
            "subungual hematoma", "syringoma", "telangiectasia", "tinea corporis", "tinea cruris",
            "tinea faciale", "tinea pedis", "tinea versicolor", "urticaria", "urticaria pigmentosa",
            "urticarial vasculitis", "varicella", "vasculitis", "venous lake", "verruca plana",
            "viral exanthem", "vitiligo", "wart", "xanthelasma", "xerotic eczema"
        ]
    },
    # Concept
    "derm7pt": {
        "task": "concept",
        "concepts": ["pigment_network", "blue_whitish_veil", "vascular_structures",
                    "pigmentation", "streaks", "dots_and_globules", "regression_structures"]
    },
    "skincon": {
        "task": "concept",
        "concepts": ["Vesicle", "Papule", "Macule", "Plaque", "Pustule", "Bulla", "Patch",
                    "Nodule", "Ulcer", "Crust", "Erosion", "Excoriation", "Atrophy",
                    "Exudate", "Fissure", "Induration", "Xerosis", "Telangiectasia",
                    "Scale", "Scar", "Friable", "Pedunculated", "Exophytic/Fungating",
                    "Warty/Papillomatous", "Dome-shaped", "Umbilicated",
                    "Brown(Hyperpigmentation)", "White(Hypopigmentation)",
                    "Purple", "Yellow", "Black", "Erythema"]
    },
    # Captioning
    "skin_cap": {"task": "captioning"},
}


def get_dataset_options(dataset: str) -> List[str]:
    """
    Get the options/diseases list for a diagnosis dataset.
    
    Args:
        dataset: Dataset name (e.g., 'SNU_500', 'HAM10000')
        
    Returns:
        List of disease options, or empty list if not found
    """
    config = DATASET_CONFIGS.get(dataset, {})
    return config.get("options", [])


def get_dataset_concepts(dataset: str) -> List[str]:
    """
    Get the concepts list for a concept annotation dataset.
    
    Args:
        dataset: Dataset name (e.g., 'derm7pt', 'skincon')
        
    Returns:
        List of concepts, or empty list if not found
    """
    config = DATASET_CONFIGS.get(dataset, {})
    return config.get("concepts", [])
