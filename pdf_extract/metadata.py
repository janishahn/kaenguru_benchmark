"""
Metadata models for extracted images from PDFs.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, field_validator
import uuid
import hashlib

def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of binary data."""
    return hashlib.sha256(data).hexdigest()

class ImageMetadata(BaseModel):
    """Metadata for an extracted image from a PDF."""
    
    source_pdf_path: str  # Changed from Path to str
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: Literal["bitmap", "vector", "raw_embedded"]
    original_ext: str
    extracted_file_path: str  # Changed from Path to str
    raw_image_data_hash: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    @field_validator('source_pdf_path', 'extracted_file_path', mode='before')
    @classmethod
    def convert_path_to_str(cls, v):
        """Convert Path objects to strings."""
        if isinstance(v, Path):
            return str(v)
        return v

    @classmethod
    def from_image_data(cls, 
                       source_pdf_path: Path,
                       page_number: int,
                       bbox: Tuple[float, float, float, float],
                       source_type: Literal["bitmap", "vector", "raw_embedded"],
                       original_ext: str,
                       extracted_file_path: Path,
                       image_data: bytes) -> 'ImageMetadata':
        """
        Create an ImageMetadata instance from image data, automatically computing the SHA256 hash.
        
        Args:
            source_pdf_path: Path to the source PDF
            page_number: Page number in the PDF (0-based)
            bbox: Bounding box (x0, y0, x1, y1)
            source_type: Type of image source
            original_ext: Original file extension
            extracted_file_path: Path where the image will be saved
            image_data: Raw image data bytes
            
        Returns:
            ImageMetadata instance with computed hash
        """
        return cls(
            source_pdf_path=source_pdf_path,
            page_number=page_number,
            bbox=bbox,
            source_type=source_type,
            original_ext=original_ext,
            extracted_file_path=extracted_file_path,
            raw_image_data_hash=compute_sha256(image_data)
        ) 