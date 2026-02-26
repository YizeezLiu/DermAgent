"""
Dataset configurations for benchmark tasks.
"""

from .base import DatasetConfig, load_dataset
from .ham10000 import HAM10000_CONFIG, HAM10000_500_CONFIG
from .snu import SNU_500_CONFIG
from .concepts import (
    DERM7PT_CONCEPT_CONFIG,
    SKINCON_CONCEPT_CONFIG,
    DERM7PT_500_CONCEPT_CONFIG,
    SKINCON_500_CONCEPT_CONFIG,
)
from .skin_cap import SKIN_CAP_CONFIG

DATASETS = {
    "HAM10000": HAM10000_CONFIG,
    "HAM10000_500": HAM10000_500_CONFIG,
    "SNU_500": SNU_500_CONFIG,
    "derm7pt_concepts": DERM7PT_CONCEPT_CONFIG,
    "skincon_concepts": SKINCON_CONCEPT_CONFIG,
    "derm7pt_500": DERM7PT_500_CONCEPT_CONFIG,
    "skincon_500": SKINCON_500_CONCEPT_CONFIG,
    "skin_cap": SKIN_CAP_CONFIG,
}

__all__ = [
    "DatasetConfig",
    "load_dataset",
    "DATASETS",
    "HAM10000_CONFIG",
    "HAM10000_500_CONFIG",
    "SNU_500_CONFIG",
    "DERM7PT_CONCEPT_CONFIG",
    "SKINCON_CONCEPT_CONFIG",
    "DERM7PT_500_CONCEPT_CONFIG",
    "SKINCON_500_CONCEPT_CONFIG",
    "SKIN_CAP_CONFIG",
]
