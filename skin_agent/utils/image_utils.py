"""
Image path utilities for anonymous ID generation.

Provides functions to:
- Generate deterministic anonymous IDs for image paths
- Register and resolve image path mappings
- Prevent filename-based data leakage in LLM prompts
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional


# Global path registry for anonymous ID to actual path mapping
_IMAGE_PATH_REGISTRY: Dict[str, str] = {}


def generate_image_id(image_path: str) -> str:
    """
    Generate an anonymous, deterministic ID for an image path.
    
    Uses MD5 hash of the absolute path to ensure:
    - Same path always produces the same ID (deterministic)
    - No disease/label information is leaked (anonymous)
    - Can be recreated for debugging if needed (traceable)
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Anonymous image ID in format "IMG_xxxxxxxx" (8 hex chars)
        
    Example:
        >>> generate_image_id("/path/to/melanoma_5.jpg")
        'IMG_a3f8c2d1'
    """
    path_str = str(Path(image_path).resolve())
    hash_digest = hashlib.md5(path_str.encode()).hexdigest()[:8]
    return f"IMG_{hash_digest}"


def register_image_path(image_id: str, actual_path: str) -> None:
    """
    Register the mapping from anonymous ID to actual path.
    
    Must be called before tools try to resolve the ID.
    
    Args:
        image_id: Anonymous image ID (e.g., "IMG_a3f8c2d1")
        actual_path: Actual file path to the image
    """
    _IMAGE_PATH_REGISTRY[image_id] = actual_path


def resolve_image_path(image_id_or_path: str) -> str:
    """
    Resolve an anonymous ID to its actual path.
    
    If the input is a regular path (not starting with "IMG_"),
    it is returned unchanged. This allows backward compatibility
    with code that still uses actual paths.
    
    Args:
        image_id_or_path: Either an anonymous ID or an actual path
        
    Returns:
        The actual file path
        
    Raises:
        ValueError: If the ID is not registered
    """
    if image_id_or_path.startswith("IMG_"):
        resolved = _IMAGE_PATH_REGISTRY.get(image_id_or_path)
        if resolved is None:
            raise ValueError(
                f"Image ID '{image_id_or_path}' not found in registry. "
                "Make sure to call register_image_path() before tool execution."
            )
        return resolved
    # Regular path, return as-is for backward compatibility
    return image_id_or_path


def get_display_id(image_path: str) -> str:
    """
    Get the display ID for an image path.
    
    If the path is already an anonymous ID, return it as-is.
    Otherwise, generate a new anonymous ID.
    
    Args:
        image_path: Either an anonymous ID or an actual path
        
    Returns:
        Anonymous image ID for display
    """
    if image_path.startswith("IMG_"):
        return image_path
    return generate_image_id(image_path)


def clear_registry() -> None:
    """
    Clear all registered image path mappings.
    
    Useful for testing or between benchmark runs.
    """
    _IMAGE_PATH_REGISTRY.clear()


def get_registry_size() -> int:
    """
    Get the number of registered image paths.
    
    Returns:
        Number of entries in the registry
    """
    return len(_IMAGE_PATH_REGISTRY)
