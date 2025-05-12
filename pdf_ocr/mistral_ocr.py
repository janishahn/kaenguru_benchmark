#!/usr/bin/env python3

import argparse
import base64
import json
import shutil
import os
import logging
import re
import subprocess
import sys
import yaml # Added import
from utils.logger_setup import setup_logging_from_config
from pathlib import Path
from typing import List, Dict, Any
from pdf_preprocessing.pdf_graphics_extractor import main as graphics_extractor_main

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

def parse_json_to_markdown_with_graphics(
    ocr_response: OCRResponse,
    graphics_metadata: Dict[str, Any],
    markdown_file_path: Path, # Absolute path to the .md file being written
    ocr_images_abs_dir: Path, # Absolute path to .../ocr_output/ocr_images/
) -> str:
    """
    Parses Mistral OCRResponse and integrates pre-extracted graphics into Markdown.

    Parameters
    ----------
    ocr_response : OCRResponse
        The OCRResponse object from the Mistral API (for the stripped PDF).
    graphics_metadata : Dict[str, Any]
        Loaded graphics metadata from the preprocessing step.
    markdown_file_path : Path
        Absolute path where the final Markdown file will be saved.
        Used to calculate relative paths for images.
    ocr_images_abs_dir : Path
        Absolute path to the directory where final images (graphics) will be stored.

    Returns
    -------
    str
        A single string containing the document text formatted in Markdown,
        with graphics integrated.
    """
    full_markdown_output_parts = []
    all_graphics_from_metadata = graphics_metadata.get('graphics', [])

    for page_idx, page_content_from_ocr in enumerate(ocr_response.pages):
        current_page_num_for_graphics = page_idx + 1  # Metadata uses 1-based page numbers
        page_specific_elements = []

        graphics_for_current_page = sorted(
            [g for g in all_graphics_from_metadata if g.get('page_num') == current_page_num_for_graphics],
            key=lambda g: g.get('bbox', [0, float('inf'), 0, 0])[1]  # Sort by y0
        )

        words_on_current_page = []
        if hasattr(page_content_from_ocr, 'words') and page_content_from_ocr.words:
            words_on_current_page = sorted(
                page_content_from_ocr.words,
                key=lambda w: (w.bbox[1], w.bbox[0])  # Sort by y0, then x0
            )

        if not words_on_current_page and hasattr(page_content_from_ocr, 'markdown') and page_content_from_ocr.markdown:
            # Case: Page has markdown from OCR but no detailed words (e.g., fully graphical page in stripped PDF)
            page_specific_elements.append(page_content_from_ocr.markdown)
            for graphic_to_insert in graphics_for_current_page:
                graphic_id = graphic_to_insert['id']
                graphic_filename = Path(graphic_to_insert['path']).name
                try:
                    # Calculate path relative from the MD file to the image in ocr_images_abs_dir
                    relative_image_path_for_md = Path(os.path.relpath(
                        ocr_images_abs_dir / graphic_filename,
                        start=markdown_file_path.parent
                    ))
                except ValueError: # Fallback for different drives on Windows, etc.
                    relative_image_path_for_md = Path(f"../{ocr_images_abs_dir.name}/{graphic_filename}")
                markdown_image_tag = f"\\n\\n![{graphic_id}]({relative_image_path_for_md.as_posix()})\\n\\n"
                page_specific_elements.append(markdown_image_tag)
        else:
            # Merge words and graphics by their y-coordinate
            word_iterator_idx = 0
            graphic_iterator_idx = 0
            current_text_line_buffer = []

            while word_iterator_idx < len(words_on_current_page) or graphic_iterator_idx < len(graphics_for_current_page):
                word_y_pos = words_on_current_page[word_iterator_idx].bbox[1] if word_iterator_idx < len(words_on_current_page) else float('inf')
                graphic_y_pos = graphics_for_current_page[graphic_iterator_idx]['bbox'][1] if graphic_iterator_idx < len(graphics_for_current_page) else float('inf')

                if word_y_pos == float('inf') and graphic_y_pos == float('inf'): # Both exhausted
                    break

                if word_y_pos <= graphic_y_pos and word_iterator_idx < len(words_on_current_page) : # Prioritize words if y is same or less
                    current_text_line_buffer.append(words_on_current_page[word_iterator_idx].text)
                    word_iterator_idx += 1
                elif graphic_iterator_idx < len(graphics_for_current_page):
                    if current_text_line_buffer:
                        page_specific_elements.append(" ".join(current_text_line_buffer))
                        current_text_line_buffer = []
                    
                    graphic_to_insert = graphics_for_current_page[graphic_iterator_idx]
                    graphic_id = graphic_to_insert['id']
                    graphic_filename = Path(graphic_to_insert['path']).name
                    try:
                        relative_image_path_for_md = Path(os.path.relpath(
                            ocr_images_abs_dir / graphic_filename,
                            start=markdown_file_path.parent
                        ))
                    except ValueError:
                        relative_image_path_for_md = Path(f"../{ocr_images_abs_dir.name}/{graphic_filename}")
                    
                    markdown_image_tag = f"\\n\\n![{graphic_id}]({relative_image_path_for_md.as_posix()})\\n\\n"
                    page_specific_elements.append(markdown_image_tag)
                    graphic_iterator_idx += 1
                else: # Should only be if words are left but word_y_pos > graphic_y_pos (which is inf)
                    # This case should be covered by the first condition if words are left.
                    # If only words are left, graphic_y_pos is inf, so word_y_pos <= graphic_y_pos is true.
                    break # Safety break

            if current_text_line_buffer: # Append any remaining text
                page_specific_elements.append(" ".join(current_text_line_buffer))

        full_markdown_output_parts.append(" ".join(page_specific_elements))
        if page_idx < len(ocr_response.pages) - 1: # Add page separator
            full_markdown_output_parts.append("\\n\\n---\\n\\n")

    return "".join(full_markdown_output_parts).strip()

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
        logging.debug(f"Saved {total_images} image(s)")
    else:
        logging.info(f"No images extracted from the document")
        
    return image_paths

def process_pdf(
    pdf_path: Path, 
    client: Mistral, 
    output_dirs: Dict[str, Path], 
    graphics_metadata_json_path: Path, 
    config: Dict[str, Any], 
    project_root: Path, 
    is_batch: bool = False
) -> bool:
    """
    Processes a single PDF file using the Mistral OCR API.

    Parameters
    ----------
    pdf_path : Path
        Path object for the input PDF file (should be the stripped PDF).
    client : Mistral
        Initialized Mistral client.
    output_dirs : dict of str to Path
        Dictionary containing Paths for json, images, and markdown output.
    graphics_metadata_json_path : Path
        Path to the graphics_metadata.json file from preprocessing for this specific PDF.
    config : Dict[str, Any]
        The loaded project configuration.
    project_root : Path
        The root directory of the project.
    is_batch : bool
        If True, this is part of a batch and detailed logs should go to file only.

    Returns
    -------
    bool
        True if processing was successful, False otherwise.
    """
    base_filename_stripped = pdf_path.stem  # e.g., "original_filename_stripped"
    original_base_filename = base_filename_stripped.replace('_stripped', '')

    json_output_path = output_dirs["json"] / f"{original_base_filename}_ocr_result.json"
    markdown_output_path = output_dirs["markdown"] / f"{original_base_filename}_ocr_result.md"
    success = True

    graphics_data = None
    if graphics_metadata_json_path and graphics_metadata_json_path.is_file():
        try:
            with open(graphics_metadata_json_path, 'r', encoding='utf-8') as f:
                graphics_data = json.load(f)
            logging.info(f"Successfully loaded graphics metadata from {graphics_metadata_json_path} for {original_base_filename}")
        except Exception as e:
            logging.warning(f"Failed to load graphics metadata from {graphics_metadata_json_path} for {original_base_filename}: {e}. Proceeding without graphics integration.")
            graphics_data = None # Ensure it's None if loading fails
    else:
        logging.info(f"No graphics metadata file provided or found at {graphics_metadata_json_path} for {original_base_filename}. Proceeding without graphics integration.")

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
            if graphics_data:
                logging.info(f"Using preprocessed graphics for {original_base_filename}. Custom markdown generation and asset copying will be performed.")
                
                # Generate markdown with integrated graphics
                markdown_content = parse_json_to_markdown_with_graphics(
                    response,
                    graphics_data,
                    markdown_output_path,       # For calculating relative paths
                    output_dirs["images"]       # Absolute path to target ocr_images dir
                )
                with open(markdown_output_path, "w", encoding="utf-8") as f_md:
                    f_md.write(markdown_content)
                logging.info(f"Markdown with integrated graphics saved to {markdown_output_path}")

                # Copy graphics assets
                # Determine base directory for all processing outputs from config
                processing_output_base_dir_cfg = config["processing_output_base_dir"]
                if Path(processing_output_base_dir_cfg).is_absolute():
                    processing_base_abs_dir = Path(processing_output_base_dir_cfg)
                else:
                    processing_base_abs_dir = project_root / processing_output_base_dir_cfg
                
                destination_images_dir = output_dirs["images"] # e.g. .../ocr_output/ocr_images/
                destination_images_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists

                copied_count = 0
                if 'graphics' in graphics_data:
                    for graphic_item in graphics_data['graphics']:
                        original_graphic_path_str = graphic_item.get('path') # Path relative to processing_base_abs_dir
                        if not original_graphic_path_str:
                            logging.warning(f"Graphic item {graphic_item.get('id')} missing 'path' in metadata for {original_base_filename}.")
                            continue
                        
                        source_file_abs = (processing_base_abs_dir / original_graphic_path_str).resolve()
                        graphic_filename = Path(original_graphic_path_str).name
                        destination_file_abs = destination_images_dir / graphic_filename
                        
                        if source_file_abs.is_file():
                            try:
                                shutil.copy2(source_file_abs, destination_file_abs)
                                copied_count += 1
                            except Exception as e_copy:
                                logging.warning(f"Failed to copy graphic {source_file_abs} to {destination_file_abs} for {original_base_filename}: {e_copy}")
                        else:
                            logging.warning(f"Source graphic file not found: {source_file_abs} for {original_base_filename}")
                    logging.info(f"Copied {copied_count} graphic assets to {destination_images_dir} for {original_base_filename}")
                else:
                    logging.warning(f"No 'graphics' key found in graphics_data for {original_base_filename}. Cannot copy assets.")

            else: # Fallback to old method if no graphics_data (Mistral handles images)
                logging.info(f"No preprocessed graphics data for {original_base_filename}. Using default Mistral image handling and markdown parsing.")
                image_paths = save_images(response, output_dirs["images"], original_base_filename) # Saves Mistral's images
                markdown_content = parse_json_to_markdown(response, image_paths) # Original function
                with open(markdown_output_path, "w", encoding="utf-8") as f_md:
                    f_md.write(markdown_content)
                logging.info(f"Markdown (default Mistral handling) saved to {markdown_output_path}")
                
        except Exception as e:
            logging.error(f"Error extracting images or generating markdown: {e}")
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
        
    return success


def main() -> None:
    """
    Main function to parse arguments and orchestrate the OCR process.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="OCR pipeline with graphics extraction preprocessing.")
    parser.add_argument("pdfs", nargs='+', help="Input PDF file(s) to process.")
    parser.add_argument("--config", required=True, help="Path to config.yaml.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        return

    # Setup logging
    setup_logging_from_config(config_path) # Uses absolute path

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) # Uses yaml import
    project_root = config_path.parent # Assuming config.yaml is at project root

    # Determine base directory for all processing outputs
    processing_output_base_dir_cfg = config["processing_output_base_dir"]
    if Path(processing_output_base_dir_cfg).is_absolute():
        processing_base_dir = Path(processing_output_base_dir_cfg)
    else:
        processing_base_dir = project_root / processing_output_base_dir_cfg
    
    # Graphics extraction specific paths (needed for locating inputs for OCR)
    graphics_extraction_cfg = config["graphics_extraction"]
    graphics_output_dir_name = graphics_extraction_cfg["output_dir_name"]
    graphics_output_abs_dir = processing_base_dir / graphics_output_dir_name # e.g., .../processing_output/graphics_pipeline_output/
    
    stripped_pdf_subdir_name = graphics_extraction_cfg["stripped_pdf_subdir"]
    # e.g., .../processing_output/graphics_pipeline_output/stripped_pdfs/
    stripped_pdf_abs_subdir = graphics_output_abs_dir / stripped_pdf_subdir_name 
    
    graphics_json_filename_from_config = graphics_extraction_cfg["graphics_json_filename"] # e.g., "graphics_metadata.json"

    # Setup output dirs for OCR, relative to processing_base_dir
    ocr_processing_cfg = config.get("ocr_processing", {})
    ocr_output_dir_name_from_config = ocr_processing_cfg.get("output_dir_name", OUTPUT_BASE_DIR_NAME) # Fallback to const
    main_ocr_output_abs_dir = processing_base_dir / ocr_output_dir_name_from_config # e.g., .../processing_output/ocr_output/
    
    output_dirs = setup_output_directories(main_ocr_output_abs_dir) # Use absolute path for OCR outputs

    # Setup Mistral client
    if not MISTRAL_AVAILABLE:
        print("Mistral API not available. Exiting.")
        return
    client = Mistral(api_key=MISTRAL_API_KEY)

    for pdf_path_str in args.pdfs:
        pdf_path = Path(pdf_path_str).resolve()
        if not pdf_path.is_file():
            logging.error(f"Input PDF not found: {pdf_path}")
            continue
            
        # --- Phase 3: Step 9 ---
        # 1. Run graphics extraction (preprocessing)
        # This assumes pdf_graphics_extractor.main processes one PDF and creates outputs
        # named after the PDF's stem within the configured directories.
        try:
            logging.info(f"Starting graphics extraction for: {pdf_path}")
            # graphics_extractor_main(str(pdf_path), str(config_path)) # Call with strings if it expects that
            # Assuming graphics_extractor_main is imported and takes Path objects:
            graphics_extractor_main(pdf_path, config_path)
            logging.info(f"Graphics extraction completed for: {pdf_path}")
        except Exception as e:
            logging.error(f"Graphics extraction failed for {pdf_path}: {e}")
            continue
        # 2. Determine path to the stripped PDF (output from graphics_extractor_main)
        stripped_pdf_path = stripped_pdf_abs_subdir / f"{pdf_path.stem}_stripped.pdf"
        if not stripped_pdf_path.is_file():
            logging.error(f"Stripped PDF not found after extraction: {stripped_pdf_path}")
            continue
        # 3. Determine path to the graphics_metadata.json for this PDF
        # Assuming it's named <pdf_stem>_<graphics_json_filename_from_config> in graphics_output_abs_dir
        current_graphics_metadata_json_path = graphics_output_abs_dir / f"{pdf_path.stem}_{graphics_json_filename_from_config}"
        
        # 4. Run OCR on stripped PDF
        logging.info(f"Processing OCR for stripped PDF: {stripped_pdf_path} using graphics metadata {current_graphics_metadata_json_path}")
        process_pdf(
            stripped_pdf_path, 
            client, 
            output_dirs, 
            current_graphics_metadata_json_path,
            config,             
            project_root,       
            is_batch=(len(args.pdfs) > 1)
        )
    logging.info("All PDFs processed.")
    print("OCR processing complete.")


if __name__ == "__main__":
    # The main() function now handles argument parsing and logging setup directly.
    main()