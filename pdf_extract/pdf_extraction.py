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
import concurrent.futures

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
    Process multiple PDF files in parallel using ProcessPoolExecutor.
    
    Args:
        pdf_paths: List of paths to PDF files
        output_dir: Directory to save extracted images and metadata
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_pdf, pdf_path, output_dir) for pdf_path in pdf_paths]
        for future in concurrent.futures.as_completed(futures):
            try:
                metadata_list = future.result()
                logger.info(f"Successfully processed PDF, extracted {len(metadata_list)} images")
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")

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