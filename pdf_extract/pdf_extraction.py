#!/usr/bin/env python3
"""
PDF Extraction Pipeline Script

This script extracts images and vector graphics from PDF files and saves them
along with metadata in a structured output directory.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add the parent directory to sys.path to allow importing from pdf_extract
sys.path.append(str(Path(__file__).parent.parent))
from pdf_extract.image_extractor import process_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_output_dir(output_dir: Path) -> Path:
    """
    Ensure the output directory exists.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        Path to the output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def process_pdfs(pdf_paths: List[Path], output_dir: Path) -> None:
    """
    Process multiple PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files
        output_dir: Directory to save extracted images and metadata
    """
    for pdf_path in pdf_paths:
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            metadata_list = process_pdf(pdf_path, output_dir)
            logger.info(f"Successfully processed {pdf_path}")
            logger.info(f"Extracted {len(metadata_list)} images")
            
            # Log summary of extracted images
            bitmap_count = sum(1 for m in metadata_list if m.source_type == "bitmap")
            vector_count = sum(1 for m in metadata_list if m.source_type == "vector")
            small_count = sum(1 for m in metadata_list if "small_artifact" in m.tags)
            
            logger.info(f"Summary for {pdf_path.name}:")
            logger.info(f"  - Bitmap images: {bitmap_count}")
            logger.info(f"  - Vector graphics: {vector_count}")
            logger.info(f"  - Small artifacts: {small_count}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            continue

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract images and vector graphics from PDF files"
    )
    parser.add_argument(
        "pdf_files",
        nargs="+",
        type=Path,
        help="One or more PDF files to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extraction_output"),
        help="Directory to save extracted images and metadata (default: extraction_output)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate PDF files
    invalid_files = [f for f in args.pdf_files if not f.exists()]
    if invalid_files:
        logger.error("The following files do not exist:")
        for f in invalid_files:
            logger.error(f"  - {f}")
        sys.exit(1)
    
    # Setup output directory
    output_dir = setup_output_dir(args.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Process PDFs
    process_pdfs(args.pdf_files, output_dir)

if __name__ == "__main__":
    main() 