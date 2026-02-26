"""
Base dataset configuration and loading utilities.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for a benchmark dataset."""
    name: str
    task: str  # e.g., "diagnosis", "segmentation", "vqa"
    csv_path: str
    data_root: str
    class_names: List[str]
    num_classes: int = field(init=False)
    
    # Column names in CSV
    filename_col: str = "filename"
    label_col: str = "label"
    split_col: str = "split"
    
    # Image directory + extension handling
    image_subdir: str = "images"
    image_ext: str = ""  # e.g., ".jpg" for HAM10000
    preserve_subdir: bool = False  # True to keep subdirectory in filename (e.g., "dataset/img.jpg")
    
    # Label handling
    label_map: Dict[str, int] = field(default_factory=dict)
    label_mode: str = "single"  # "single" | "multilabel" | "captioning" | "vqa"
    label_cols: Optional[List[str]] = None  # required when label_mode == multilabel
    
    # Prompt/Q&A configuration
    prompt_templates: List[str] = field(default_factory=list)
    vqa_prompt: str = ""
    question_col: str = "question"
    question_type_col: str = "question_type"
    option_cols: List[str] = field(default_factory=lambda: ["option_A", "option_B", "option_C", "option_D"])
    
    def __post_init__(self):
        self.num_classes = len(self.class_names)
        if self.label_mode not in {"single", "multilabel", "captioning", "vqa"}:
            raise ValueError(f"Unsupported label_mode: {self.label_mode}")
        if self.label_mode == "multilabel":
            if not self.label_cols:
                self.label_cols = self.class_names
            missing = set(self.label_cols) - set(self.class_names)
            if missing:
                raise ValueError(
                    f"Label columns {missing} missing from class_names for dataset {self.name}"
                )


def load_dataset(
    config: DatasetConfig,
    split: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    Load dataset from CSV and build image paths.
    
    Args:
        config: Dataset configuration
        split: Optional split to filter (train/val/test)
        max_samples: Optional limit on number of samples
        
    Returns:
        df: DataFrame with metadata
        image_paths: List of full image paths
        labels: List of ground truth labels
    """
    print(f"\nLoading dataset: {config.name}")
    print(f"CSV: {config.csv_path}")
    
    df = pd.read_csv(config.csv_path)
    print(f"Total samples: {len(df)}")
    
    # Filter by split
    if split and config.split_col in df.columns:
        df = df[df[config.split_col] == split]
        print(f"Samples in '{split}' split: {len(df)}")
    
    # Limit samples
    if max_samples:
        df = df.head(max_samples)
        print(f"Limited to {len(df)} samples")
    
    # Build image paths
    image_paths = []
    for filename in df[config.filename_col]:
        # Preserve subdirectory structure if configured, otherwise use base filename
        if config.preserve_subdir:
            img_name = str(filename)
        else:
            img_name = Path(filename).name
        # Add extension if specified (e.g., for HAM10000 where image_id doesn't include .jpg)
        if config.image_ext and not img_name.endswith(config.image_ext):
            img_name = img_name + config.image_ext
        full_path = Path(config.data_root) / config.image_subdir / img_name
        image_paths.append(str(full_path))
    
    # Check for missing images
    missing = sum(1 for p in image_paths if not Path(p).exists())
    if missing > 0:
        print(f"Warning: {missing} images not found")
    
    # Get labels (apply mapping if provided)
    if config.label_mode == "multilabel":
        if not config.label_cols:
            raise ValueError(f"Dataset {config.name} missing label_cols for multilabel mode")
        labels = df[config.label_cols].astype(int).values
    elif config.label_mode in {"captioning", "vqa"}:
        labels = df[config.label_col].astype(str).values.tolist()
    else:
        raw_labels = df[config.label_col].values.tolist()
        if config.label_map:
            labels = [config.label_map.get(str(l), -1) for l in raw_labels]
            unknown = sum(1 for l in labels if l == -1)
            if unknown > 0:
                print(f"Warning: {unknown} unknown labels")
        else:
            labels = raw_labels
    
    return df, image_paths, labels

