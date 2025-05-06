#!/usr/bin/env python3

import argparse
import base64
import json
import os
import logging
import re
from pathlib import Path
from typing import List, Dict

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Progress bar will not be shown. Install with 'pip install tqdm'")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. .env file will not be loaded.")

try:
    from mistralai import Mistral
    from mistralai.models import OCRResponse
    MISTRAL_AVAILABLE = True
except ImportError:
    print("Error: The 'mistralai' library is required. Please install it using 'pip install mistralai'")
    MISTRAL_AVAILABLE = False

# --- Configuration ---

if DOTENV_AVAILABLE:
    load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Define output directory names relative to the project root
OUTPUT_BASE_DIR_NAME = "ocr_output"
JSON_OUTPUT_DIR_NAME = "ocr_json"
IMAGE_OUTPUT_DIR_NAME = "ocr_images"
MARKDOWN_OUTPUT_DIR_NAME = "ocr_markdown"
LOG_DIR_NAME = "logs"

# --- Helper Functions ---

def setup_logging(main_output_dir: Path, verbose: bool = False) -> None:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    main_output_dir : Path
        The base directory where logs will be stored
    verbose : bool
        If True, set console output to DEBUG level
    """
    log_dir = main_output_dir / LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "ocr_processing.log"
    
    # Configure logging: only file handler, no console output
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file)
        ]
    )

def setup_output_directories(main_output_dir: Path) -> Dict[str, Path]:
    """
    Creates the necessary output directories if they don't exist.

    Parameters
    ----------
    main_output_dir : Path
        The base directory where output folders will be created.

    Returns
    -------
    dict of str to Path
        Dictionary containing paths for json, images, and markdown output directories.
    """
    dirs = {
        "json": main_output_dir / JSON_OUTPUT_DIR_NAME,
        "images": main_output_dir / IMAGE_OUTPUT_DIR_NAME,
        "markdown": main_output_dir / MARKDOWN_OUTPUT_DIR_NAME,
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs

def format_image_filename(base_filename: str, page_idx: int, img_idx: int, ext: str) -> str:
    """Format image filename consistently across the script."""
    return f"{base_filename}_page_{page_idx+1}_img_{img_idx+1}.{ext}"

def parse_json_to_markdown(response: OCRResponse, image_paths: Dict[int, Dict[int, str]]) -> str:
    """
    Parses the Mistral OCRResponse JSON structure into a continuous Markdown string.

    Parameters
    ----------
    response : OCRResponse
        The OCRResponse object from the Mistral API.
    image_paths : Dict[int, Dict[int, str]]
        Dictionary mapping page and image indices to their saved paths.

    Returns
    -------
    str
        A single string containing the document text formatted in Markdown.
    """
    if not response or not response.pages:
        return ""
        
    markdown_parts = []
    
    # Create a flat list of all images from all pages for continuous numbering
    all_images = []
    for page_idx in sorted(image_paths.keys()):
        for img_idx in sorted(image_paths[page_idx].keys()):
            all_images.append({
                'page_idx': page_idx,
                'img_idx': img_idx,
                'path': image_paths[page_idx][img_idx]
            })
    
    # Process each page's markdown
    for page_idx, page in enumerate(response.pages):
        if not hasattr(page, 'markdown') or not page.markdown:
            continue
            
        markdown_content = page.markdown
        
        # Replace image references if we have saved images for this page
        if page_idx in image_paths:
            # Find all markdown image references
            md_img_matches = list(re.finditer(r'!\[.*?\]\([^)]+\)', markdown_content))
            
            # Find all HTML image tags
            html_img_matches = list(re.finditer(r'<img[^>]+src=["\'][^"\']+["\']', markdown_content))
            
            # Combined mapping between image index and position in markdown
            img_replacements = []
            
            # Process markdown image syntax
            for match in md_img_matches:
                img_replacements.append({
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'markdown'
                })
            
            # Process HTML image tags
            for match in html_img_matches:
                img_replacements.append({
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'html'
                })
            
            # Sort by position in the document
            img_replacements.sort(key=lambda x: x['start'])
            
            # Apply replacements from end to start to preserve positions
            for replacement_idx, replacement in enumerate(reversed(img_replacements)):
                # Find which image from this page we're processing
                avail_indices = sorted(image_paths[page_idx].keys(), reverse=True)
                if replacement_idx < len(avail_indices):
                    local_img_idx = avail_indices[replacement_idx]
                    img_path = image_paths[page_idx][local_img_idx]
                    
                    # Get the continuous image number from our mapping
                    img_number = None
                    for img in all_images:
                        if img['page_idx'] == page_idx and img['img_idx'] == local_img_idx:
                            # Find the position (1-based index) of this image in the all_images list
                            img_number = all_images.index(img) + 1
                            break
                    
                    if img_number is None:
                        # Fallback if not found
                        img_number = replacement_idx + 1
                    
                    # Extract the portion of text to replace
                    orig_text = markdown_content[replacement['start']:replacement['end']]
                    
                    # Create the replacement text
                    if replacement['type'] == 'markdown':
                        new_text = f'![Image {img_number}]({img_path})'
                    else:  # HTML
                        new_text = re.sub(r'src=["\'][^"\']+["\']', f'src="{img_path}"', orig_text)
                    
                    # Perform the replacement
                    markdown_content = markdown_content[:replacement['start']] + new_text + markdown_content[replacement['end']:]
        
        markdown_parts.append(markdown_content)
    
    full_markdown = "\n\n".join(markdown_parts).strip()
    return full_markdown

def save_images(response: OCRResponse, output_dir: Path, base_filename: str) -> Dict[int, Dict[int, str]]:
    """
    Extracts and saves images from the OCR response.
    
    Returns
    -------
    Dict[int, Dict[int, str]]
        Dictionary mapping page indices to image indices and their relative paths
    """
    if not response or not response.pages:
        logging.info(f"No pages found in response for '{base_filename}'.")
        return {}

    total_images = 0
    image_paths = {}  # {page_idx: {img_idx: relative_path}}

    for page_idx, page in enumerate(response.pages):
        if not hasattr(page, 'images') or not page.images:
            continue
            
        page_images = {}
        for img_idx, image in enumerate(page.images):
            try:
                img_b64 = image.image_base64
                if not img_b64:
                    continue

                # Detect and strip data URL prefix
                ext = "png"  # default
                if img_b64.startswith("data:image/"):
                    header, b64data = img_b64.split(",", 1)
                    if "jpeg" in header or "jpg" in header:
                        ext = "jpg"
                    elif "png" in header:
                        ext = "png"
                    img_b64 = b64data
                else:
                    # fallback: no header, assume png
                    ext = "png"

                image_bytes = base64.b64decode(img_b64)
                image_filename = format_image_filename(base_filename, page_idx, img_idx, ext)
                image_path = output_dir / image_filename
                relative_path = f"../ocr_images/{image_filename}"  # Path relative to markdown file
                
                with open(image_path, "wb") as f_img:
                    f_img.write(image_bytes)
                total_images += 1
                page_images[img_idx] = relative_path
                
            except Exception as e:
                logging.error(f"Error processing image {img_idx+1} on page {page_idx+1}: {e}")
                
        if page_images:
            image_paths[page_idx] = page_images

    if total_images > 0:
        logging.info(f"Saved {total_images} image(s)")
    else:
        logging.info(f"No images extracted from the document")
        
    return image_paths

def process_pdf(pdf_path: Path, client: Mistral, output_dirs: Dict[str, Path], is_batch: bool = False) -> bool:
    """
    Processes a single PDF file using the Mistral OCR API.

    Parameters
    ----------
    pdf_path : Path
        Path object for the input PDF file.
    client : Mistral
        Initialized Mistral client.
    output_dirs : dict of str to Path
        Dictionary containing Paths for json, images, and markdown output.
    is_batch : bool
        If True, this is part of a batch and detailed logs should go to file only

    Returns
    -------
    bool
        True if processing was successful, False otherwise
    """
    # Set console log level for this function
    if is_batch:
        console_handler = None
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                console_handler = handler
                handler.setLevel(logging.INFO)  # Only show important messages on console
    
    if not is_batch:
        logging.info(f"\nProcessing '{pdf_path.name}'...")
    
    base_filename = pdf_path.stem
    json_output_path = output_dirs["json"] / f"{base_filename}_ocr_result.json"
    markdown_output_path = output_dirs["markdown"] / f"{base_filename}_ocr_result.md"
    success = True

    try:
        with open(pdf_path, "rb") as pdf_file:
            logging.debug(f"Uploading file to Mistral API: {pdf_path.name}")
            upload_resp = client.files.upload(
                file={"file_name": pdf_path.name, "content": pdf_file},
                purpose="ocr"
            )
            file_id = upload_resp.id
            logging.debug(f"Uploaded file ID: {file_id}")

            file_info = client.files.retrieve(file_id=file_id)
            logging.debug(f"File info: {file_info}")

            signed_url = client.files.get_signed_url(file_id=file_id)
            logging.debug(f"Signed URL obtained")

            logging.debug(f"Processing with Mistral OCR...")
            response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
                include_image_base64=True
            )
            logging.debug(f"OCR processing complete")

            try:
                client.files.delete(file_id=file_id)
                logging.debug(f"Cleaned up file ID: {file_id}")
            except Exception as e:
                logging.warning(f"Failed to delete file ID {file_id}: {e}")

        try:
            response_dict = response.model_dump(mode='json')
            with open(json_output_path, "w", encoding="utf-8") as f_json:
                json.dump(response_dict, f_json, indent=2, ensure_ascii=False)
            logging.debug(f"Saved raw JSON response")
        except (IOError, TypeError) as e:
            logging.error(f"Error saving JSON response: {e}")
            success = False

        # Extract and save images first to get image paths
        try:
            image_paths = save_images(response, output_dirs["images"], base_filename)
        except Exception as e:
            logging.error(f"Error extracting images: {e}")
            image_paths = {}
            success = False

        # Extract markdown with updated image paths
        try:
            markdown_content = parse_json_to_markdown(response, image_paths)
            with open(markdown_output_path, "w", encoding="utf-8") as f_md:
                f_md.write(markdown_content)
            logging.debug(f"Saved Markdown output")
        except Exception as e:
            logging.error(f"Error generating or saving Markdown: {e}")
            success = False

    except Exception as e:
        logging.error(f"Mistral API error: {e}")
        if 'file_id' in locals():
            try:
                client.files.delete(file_id=file_id)
                logging.debug(f"Cleaned up file ID: {file_id} after error")
            except Exception:
                pass
        success = False

    # Restore console log level if this was a batch operation
    if is_batch and console_handler:
        console_handler.setLevel(logging.INFO)
        
    return success


def main():
    """
    Main function to parse arguments and orchestrate the OCR process.

    Returns
    -------
    None
    """
    if not MISTRAL_AVAILABLE:
        print("Error: The 'mistralai' library is required. Please install it using 'pip install mistralai'")
        exit(1)

    parser = argparse.ArgumentParser(
        description="Perform OCR on PDF files using the Mistral AI API."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Relative path to a single PDF file or a directory containing PDF files.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output to console (always enabled for logs)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    project_root_dir = script_dir.parent
    main_output_dir = project_root_dir / OUTPUT_BASE_DIR_NAME
    
    # Setup logging
    setup_logging(main_output_dir, args.verbose)
    
    logging.info(f"Project Root Directory: {project_root_dir}")
    logging.info(f"Output will be saved under: {main_output_dir}")

    input_path_arg = Path(args.input_path)
    input_path = input_path_arg.resolve()

    if not input_path.exists():
        logging.error(f"Input path '{args.input_path}' (resolved to '{input_path}') does not exist.")
        exit(1)

    pdf_files_to_process: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            pdf_files_to_process.append(input_path)
        else:
            logging.error(f"Input file '{args.input_path}' is not a PDF file.")
            exit(1)
    elif input_path.is_dir():
        logging.info(f"Scanning directory '{input_path}' for PDF files...")
        pdf_files_to_process = sorted(list(input_path.glob("*.pdf")))
        if not pdf_files_to_process:
            logging.info(f"No PDF files found in directory '{input_path}'.")
            exit(0)
        logging.info(f"Found {len(pdf_files_to_process)} PDF file(s).")
    else:
        logging.error(f"Input path '{args.input_path}' is neither a file nor a directory.")
        exit(1)

    output_dirs = setup_output_directories(main_output_dir)
    logging.info("Output directories ensured.")

    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        logging.info("Mistral client initialized.")
    except Exception as e:
        logging.error(f"Error initializing Mistral client: {e}")
        exit(1)

    # Determine if we're batch processing or single file
    is_batch = len(pdf_files_to_process) > 1
    
    # Process files
    if is_batch and TQDM_AVAILABLE:
        # Use tqdm for progress tracking with multiple files
        successes = 0
        total = len(pdf_files_to_process)
        with tqdm(total=total, desc="Processing PDFs", unit="file") as pbar:
            for pdf_path in pdf_files_to_process:
                logging.info(f"Processing ({pbar.n+1}/{total}): {pdf_path.name}")
                if process_pdf(pdf_path, client, output_dirs, is_batch=True):
                    successes += 1
                pbar.update(1)
        
        logging.info(f"Successfully processed {successes}/{total} files")
    else:
        # Process files without tqdm
        for pdf_path in pdf_files_to_process:
            process_pdf(pdf_path, client, output_dirs, is_batch=is_batch)

    logging.info("\nOCR processing finished.")
    print("OCR processing complete.")


if __name__ == "__main__":
    main()