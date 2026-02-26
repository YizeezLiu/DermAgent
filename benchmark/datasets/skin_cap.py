"""
SkinCap Dataset Configuration for Captioning Task.
"""

from .base import DatasetConfig

SKIN_CAP_CONFIG = DatasetConfig(
    name="skin_cap",
    task="captioning",
    csv_path="datasets/skin_cap/skin_cap_meta_100.csv",
    data_root="datasets/skin_cap",
    image_subdir="images",
    class_names=[],  # No class names for captioning
    filename_col="filename",
    label_col="caption_zh_polish_en",
    label_mode="captioning",
    vqa_prompt="Describe the skin lesion in this image in detail, including its color, shape, borders, and any specific dermatological features."
)



















