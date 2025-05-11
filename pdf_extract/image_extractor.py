"""
Image extraction module for PDF processing.
Handles extraction of both bitmap and vector images from PDF files.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import fitz  # PyMuPDF
import logging
import os
import json
import statistics
from .metadata import ImageMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
MIN_IMAGE_AREA = 30 * 30  # Reduced from 50x50 to 30x30 pixels
MIN_VECTOR_PATH_COMPLEXITY = 2  # Reduced from 3 to 2 items
MIN_VECTOR_AREA = 40 * 40  # Reduced from 75x75 to 40x40 pixels
IOU_THRESHOLD = 0.3  # Intersection over Union threshold for overlap detection
MIN_COLOR_VARIANCE = 0.05  # Minimum variance in pixel values to filter empty images
MAX_PATH_DISTANCE = 5  # Increased from 1 to allow grouping across small gaps (e.g., between plank and insect)
TOUCHING_DISTANCE = 5  # Increased from 1 to allow consolidation across small gaps
BORDER_PADDING = 2  # Padding for final output
MIN_LONG_DIMENSION_FOR_LINES = 100 # Minimum long dimension for a line-like graphic
MAX_SHORT_DIMENSION_FOR_LINES = 15 # Maximum short dimension for a line-like graphic

def calculate_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection rectangle
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    
    # Check if boxes overlap
    if x1 <= x0 or y1 <= y0:
        return 0.0
        
    # Calculate areas
    intersection_area = (x1 - x0) * (y1 - y0)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area
    
    return intersection_area / union_area

def check_overlapping_boxes(metadata_list: List[ImageMetadata]) -> None:
    """
    Check for overlapping bounding boxes and log warnings.
    
    Args:
        metadata_list: List of ImageMetadata objects to check
    """
    for i, meta1 in enumerate(metadata_list):
        for j, meta2 in enumerate(metadata_list[i+1:], i+1):
            # Only check boxes on the same page
            if meta1.page_number != meta2.page_number:
                continue
                
            iou = calculate_iou(meta1.bbox, meta2.bbox)
            if iou > IOU_THRESHOLD:
                logger.warning(
                    f"Significant overlap detected between images on page {meta1.page_number}: "
                    f"Image {meta1.image_id} ({meta1.source_type}) and "
                    f"Image {meta2.image_id} ({meta2.source_type}) "
                    f"with IoU {iou:.2f}"
                )

def check_content_emptiness(file_path: Path) -> bool:
    """
    Check if an image file contains actual visual content or is essentially empty.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        True if the image appears to be empty, False otherwise
    """
    try:
        # Open the image as a pixmap
        pixmap = fitz.Pixmap(file_path)
        
        # For color images, check variance in color channels
        samples = list(pixmap.samples)
        
        # If image is too small, consider it an artifact
        if pixmap.width < 20 or pixmap.height < 20:
            return True
            
        # Calculate variance of pixel values to detect empty/uniform images
        if len(samples) > 0:
            # Get samples with proper stride based on color mode
            stride = pixmap.n
            samples_by_channel = [samples[i::stride] for i in range(stride)]
            
            # Calculate variance for each channel
            variances = [statistics.variance(channel) if len(channel) > 1 else 0 
                         for channel in samples_by_channel]
            
            # Image is empty if variance is very low across all channels
            if max(variances) < MIN_COLOR_VARIANCE:
                return True
                
        return False
    except Exception as e:
        logger.warning(f"Error checking image content for {file_path}: {str(e)}")
        return False
    finally:
        if 'pixmap' in locals():
            pixmap = None  # Pixmap is freed when reference is gone

def check_small_images(metadata_list: List[ImageMetadata], min_area: int = MIN_IMAGE_AREA) -> List[ImageMetadata]:
    """
    Check for small images and empty content, marking or filtering them.
    
    Args:
        metadata_list: List of ImageMetadata objects to check
        min_area: Minimum area in pixels to consider an image as small
        
    Returns:
        Filtered list of ImageMetadata objects
    """
    filtered_list = []
    
    for metadata in metadata_list:
        width = metadata.bbox[2] - metadata.bbox[0]
        height = metadata.bbox[3] - metadata.bbox[1]
        area = width * height
        threshold = MIN_VECTOR_AREA if metadata.source_type == "vector" else min_area

        if area < threshold:
            metadata.tags.append("small_artifact")
            logger.debug(f"Marked image {metadata.image_id} as small artifact (area: {area:.1f} pixels) on page {metadata.page_number}")
            if metadata.extracted_file_path:
                # Use a local Path object for unlink without modifying the metadata field
                path_obj = Path(metadata.extracted_file_path) if isinstance(metadata.extracted_file_path, str) else metadata.extracted_file_path
                if not path_obj.exists() and isinstance(metadata.extracted_file_path, str):
                    logger.warning(f"extracted_file_path for {metadata.image_id} does not exist: '{metadata.extracted_file_path}'")
                elif path_obj.exists():
                    try:
                        path_obj.unlink()
                    except OSError as e:
                        logger.error(f"Error unlinking small artifact file {path_obj}: {e}")
            continue
            
        if metadata.extracted_file_path:
            # Use a local Path object for content emptiness check and unlink without modifying the metadata field
            path_obj = Path(metadata.extracted_file_path) if isinstance(metadata.extracted_file_path, str) else metadata.extracted_file_path
            if path_obj.exists() and check_content_emptiness(path_obj):
                metadata.tags.append("empty_content")
                logger.debug(f"Marked image {metadata.image_id} as empty content on page {metadata.page_number}")
                try:
                    path_obj.unlink()
                except OSError as e:
                    logger.error(f"Error unlinking empty content file {path_obj}: {e}")
                continue
            elif not path_obj.exists():
                logger.warning(f"File {path_obj} for image {metadata.image_id} (vector) does not exist before content check.")

        filtered_list.append(metadata)
    
    return filtered_list

def ensure_output_dir(output_dir: Path, pdf_name: str, page_num: int) -> Path:
    """
    Ensure the output directory structure exists for a specific page.
    
    Args:
        output_dir: Base output directory
        pdf_name: Name of the PDF file
        page_num: Page number
        
    Returns:
        Path to the page-specific output directory
    """
    page_dir = output_dir / pdf_name / f"page_{page_num}"
    page_dir.mkdir(parents=True, exist_ok=True)
    return page_dir

def get_image_extension(image_info: dict) -> str:
    """
    Get the appropriate file extension for an image based on its format.
    
    Args:
        image_info: Image information dictionary from PyMuPDF
        
    Returns:
        File extension (e.g., 'jpg', 'png')
    """
    ext = image_info.get('ext', '').lower()
    if not ext:
        # Default to png if no extension is found
        return 'png'
    return ext

def is_text_path(path: Dict) -> bool:
    """
    Determine if a path is likely a text rendering path.
    
    Args:
        path: Path dictionary from PyMuPDF
        
    Returns:
        True if the path appears to be text, False otherwise
    """
    # Get the rectangle early on so we can calculate width/height whenever needed
    rect = path.get('rect')
    # Define width and height variables with defaults to avoid undefined variable issues
    width, height = 0, 0
    if rect:
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
    
    # Special handling for known graphic elements: paths with both fill and stroke
    # that are rectangular or have specific characteristics of figures
    has_stroke = path.get('stroke_color') is not None and path.get('stroke_width', 0) > 0
    has_fill = path.get('fill') is not None or path.get('fill_color') is not None
    
    # If it's a rectangle with both fill and stroke, consider it a graphic (like a box with content)
    if rect and has_stroke and has_fill:
        # If it's a reasonably sized box with both fill and stroke, treat as graphic
        if width > 20 and height > 20 and width < 300 and height < 300:
            return False
    
    # Check for paths with "handle" in path.get('items', []) - this can capture vector graphics elements
    items = path.get('items', [])
    if len(items) > 0:
        for item in items:
            if isinstance(item, list) and len(item) > 0:
                if item[0] in ['l', 'c', 'v', 'y']:  # Line, curve, etc. commands in vector paths
                    # If multiple path commands and reasonable size, likely graphic
                    if len(items) >= 3:
                        return False
    
    # Original checks (now less strict)
    if path.get('type') == 'text':
        return True
    
    # Check for very small, simple paths that are likely text
    if len(path.get('items', [])) < 2:  # Only filter out extremely simple paths
        # But give special treatment to rectangles or filled shapes
        if has_fill and rect and width > 10 and height > 10:
            return False
        return True
        
    # Check if the path forms a very small rectangle (likely decorative elements)
    if rect:
        # Reduce when we consider something to be too small and likely text/decoration
        if width < 3 or height < 3 or (width * height < 30):
            return True
    
    # Special case for straight horizontal or vertical lines that might be part of graphics
    # like the plank in the ant/ladybug example
    if items and len(items) >= 2:
        horizontal_line = True
        vertical_line = True
        start_y = None
        start_x = None
        
        for item in items:
            if isinstance(item, list) and len(item) >= 3:
                if item[0] == 'm':  # Move command
                    start_x, start_y = item[1], item[2]
                elif item[0] == 'l':  # Line command
                    # Check if it maintains same y (horizontal) or same x (vertical)
                    if start_y is not None and abs(item[2] - start_y) > 1:
                        horizontal_line = False
                    if start_x is not None and abs(item[1] - start_x) > 1:
                        vertical_line = False
                    start_x, start_y = item[1], item[2]
                else:
                    horizontal_line = False
                    vertical_line = False
        
        # If it's a simple horizontal or vertical line with reasonable length, treat as graphic
        if (horizontal_line or vertical_line) and rect:
            if (horizontal_line and width > 50) or (vertical_line and height > 50):
                return False
    
    # Check for typical text-related stroke colors - less aggressive
    stroke_color = path.get('stroke_color')
    if stroke_color in [(0, 0, 0), (0, 0, 0, 1)]:  # Black stroke is often text
        # Only count as text if both very simple AND black - more stringent
        if len(path.get('items', [])) < 2 and not has_fill:
            return True
            
    return False

def add_padding_to_bbox(bbox: Tuple[float, float, float, float], padding: float = BORDER_PADDING) -> Tuple[float, float, float, float]:
    """
    Add padding to a bounding box.
    
    Args:
        bbox: Bounding box (x0, y0, x1, y1)
        padding: Amount of padding to add
        
    Returns:
        Padded bounding box
    """
    return (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)

def paths_are_related(path1: Dict, path2: Dict) -> bool:
    """
    Determine if two raw paths are likely part of the same initial drawing element.
    Uses unpadded bounding boxes.
    """
    rect1 = fitz.Rect(path1["rect"])  # Original bounding box for path1
    rect2 = fitz.Rect(path2["rect"])  # Original bounding box for path2

    area1 = rect1.width * rect1.height
    area2 = rect2.width * rect2.height

    # 1. Check for direct intersection of original bounding boxes
    if rect1.intersects(rect2):
        # If paths intersect, consider them related.
        # The previous size ratio check (e.g., max_area / min_area > 250) was too restrictive
        # for small details on larger objects (like the ant/ladybug on the plank).
        # Also, if one path has zero area (e.g., a line) and it intersects another, they are related.
        return True

    # 2. If not intersecting, check for closeness and alignment
    # Calculate horizontal and vertical gaps between the *original* bounding boxes
    h_gap = -1.0
    if rect1.x1 < rect2.x0:  # rect1 is to the left of rect2
        h_gap = rect2.x0 - rect1.x1
    elif rect2.x1 < rect1.x0:  # rect2 is to the left of rect1
        h_gap = rect1.x0 - rect2.x1

    if h_gap != -1.0 and h_gap <= MAX_PATH_DISTANCE:
        if max(rect1.y0, rect2.y0) < min(rect1.y1, rect2.y1):
            if area1 == 0 or area2 == 0: return True
            if min(area1, area2) > 0 and max(area1, area2) / min(area1, area2) > 5:
                return False
            return True

    v_gap = -1.0
    if rect1.y1 < rect2.y0:  # rect1 is above rect2
        v_gap = rect2.y0 - rect1.y1
    elif rect2.y1 < rect1.y0:  # rect2 is above rect1
        v_gap = rect1.y0 - rect2.y1
    
    if v_gap != -1.0 and v_gap <= MAX_PATH_DISTANCE:
        if max(rect1.x0, rect2.x0) < min(rect1.x1, rect2.x1):
            if area1 == 0 or area2 == 0: return True
            if min(area1, area2) > 0 and max(area1, area2) / min(area1, area2) > 5:
                return False
            return True
            
    return False

def extract_bitmap_images(pdf_path: Path, output_dir: Path) -> List[ImageMetadata]:
    """
    Extract bitmap images from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        
    Returns:
        List of ImageMetadata objects for extracted images
    """
    logger.info(f"Extracting bitmap images from {pdf_path}")
    
    # Get PDF name without extension for output directory
    pdf_name = pdf_path.stem
    metadata_list = []
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get all images on the page
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                try:
                    # Get the image's xref (cross-reference number)
                    xref = img_info[0]
                    
                    # Get the base image
                    base_image = doc.extract_image(xref)
                    
                    if base_image:
                        # Get image data and extension
                        image_bytes = base_image["image"]
                        ext = get_image_extension(base_image)
                        
                        # Get image rectangle (bounding box)
                        bbox = page.get_image_bbox(img_info)
                        
                        # Calculate image size
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        
                        # Skip very small images directly
                        if width * height < MIN_IMAGE_AREA:
                            logger.debug(f"Skipping small image {img_index} on page {page_num} (size: {width}x{height})")
                            continue
                        
                        # Skip very large images (likely backgrounds or decorative elements)
                        if width > 500 or height > 500:
                            logger.debug(f"Skipping large image {img_index} on page {page_num} (size: {width}x{height})")
                            continue
                        
                        # Create output directory for this page
                        page_dir = ensure_output_dir(output_dir, pdf_name, page_num)
                        
                        # Create output file path
                        image_id = f"img_{img_index}"
                        output_path = page_dir / f"{image_id}.{ext}"
                        
                        # Save the raw image bytes
                        with open(output_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # Create metadata
                        metadata = ImageMetadata.from_image_data(
                            source_pdf_path=pdf_path,
                            page_number=page_num,
                            bbox=bbox,
                            source_type="bitmap",
                            original_ext=ext,
                            extracted_file_path=output_path,
                            image_data=image_bytes
                        )
                        
                        metadata_list.append(metadata)
                        logger.debug(f"Extracted image {image_id} from page {page_num}")
                        
                except Exception as e:
                    logger.error(f"Error extracting image {img_index} from page {page_num}: {str(e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()
            
    logger.info(f"Extracted {len(metadata_list)} bitmap images from {pdf_path}")
    return metadata_list

def is_contained(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> bool:
    """
    Check if one bounding box is completely contained within another.
    
    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)
        
    Returns:
        True if bbox1 is contained within bbox2, False otherwise
    """
    return (bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and 
            bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3])

def are_touching(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float], max_distance: float = TOUCHING_DISTANCE) -> bool:
    """
    Check if two bounding boxes are touching or very close to each other.
    
    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)
        max_distance: Maximum allowed distance between boxes
        
    Returns:
        True if boxes are touching or very close, False otherwise
    """
    # Check if boxes overlap
    if (bbox1[0] <= bbox2[2] and bbox1[2] >= bbox2[0] and
        bbox1[1] <= bbox2[3] and bbox1[3] >= bbox2[1]):
        return True
        
    # Check horizontal distance
    h_dist = min(abs(bbox1[0] - bbox2[2]), abs(bbox1[2] - bbox2[0]))
    if h_dist <= max_distance:
        # Check if boxes are vertically aligned
        if (bbox1[1] <= bbox2[3] and bbox1[3] >= bbox2[1]):
            return True
            
    # Check vertical distance
    v_dist = min(abs(bbox1[1] - bbox2[3]), abs(bbox1[3] - bbox2[1]))
    if v_dist <= max_distance:
        # Check if boxes are horizontally aligned
        if (bbox1[0] <= bbox2[2] and bbox1[2] >= bbox2[0]):
            return True
            
    return False

def consolidate_graphics(graphics_list: List[Tuple[List[Dict], fitz.Rect]]) -> List[Tuple[List[Dict], fitz.Rect]]:
    """
    Consolidate graphics using a two-step process:
    1. Remove graphics that are completely contained in others
    2. Group graphics that are touching or very close
    
    Args:
        graphics_list: List of (paths, bbox) tuples
        
    Returns:
        Consolidated list of (paths, bbox) tuples
    """
    if not graphics_list:
        return []
        
    # Step 1: Remove contained graphics
    filtered_graphics_step1 = []
    contained_indices = set()
    
    # Sort by area (largest first)
    sorted_graphics = sorted(
        graphics_list,
        key=lambda x: x[1].width * x[1].height,
        reverse=True
    )
    
    for i, (paths1, bbox1) in enumerate(sorted_graphics):
        if i in contained_indices:
            continue
            
        is_bbox_contained_flag = False # Renamed to avoid conflict with function
        for j, (paths2, bbox2) in enumerate(sorted_graphics):
            if j >= i or j in contained_indices: # Check against already processed or smaller items
                continue
                
            if is_contained(bbox1, bbox2): # Calls global is_contained function
                is_bbox_contained_flag = True
                contained_indices.add(i)
                logger.debug(f"Removed contained graphic (size: {bbox1.width:.1f}x{bbox1.height:.1f} because it is in {bbox2.width:.1f}x{bbox2.height:.1f})")
                break
                
        if not is_bbox_contained_flag:
            filtered_graphics_step1.append((paths1, bbox1))
            
    # Step 2: Group touching graphics
    final_consolidated_graphics = []
    processed_indices_step2 = set()
    
    # Iterate over graphics filtered in step 1
    for i in range(len(filtered_graphics_step1)):
        if i in processed_indices_step2:
            continue
            
        paths1_outer, bbox1_outer = filtered_graphics_step1[i]
        current_group_paths = list(paths1_outer)
        current_group_bbox = fitz.Rect(bbox1_outer)
        processed_indices_step2.add(i)
        
        # Look for touching graphics among the remaining items
        for j in range(i + 1, len(filtered_graphics_step1)):
            if j in processed_indices_step2:
                continue
            
            paths2_inner, bbox2_inner = filtered_graphics_step1[j]
            
            # Check if the current_group_bbox touches bbox2_inner
            if are_touching(current_group_bbox, bbox2_inner):
                current_group_paths.extend(paths2_inner)
                current_group_bbox.include_rect(bbox2_inner) # Grow the current group's bbox
                processed_indices_step2.add(j)
                logger.debug(f"Consolidated touching graphics (current group: {current_group_bbox.width:.1f}x{current_group_bbox.height:.1f} with {bbox2_inner.width:.1f}x{bbox2_inner.height:.1f})")
                
        final_consolidated_graphics.append((current_group_paths, current_group_bbox))
        
    return final_consolidated_graphics

def extract_vector_graphics(pdf_path: Path, output_dir: Path, page_num: int) -> List[ImageMetadata]:
    """
    Extract vector graphics from a specific page of a PDF file by rasterizing their bounding boxes.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images
        page_num: Page number to extract from (0-based)
        
    Returns:
        List of ImageMetadata objects for extracted vector graphics
    """
    logger.info(f"Extracting vector graphics from page {page_num} of {pdf_path}")
    
    metadata_list = []
    pdf_name = pdf_path.stem
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Get all drawings from the page
        drawings = page.get_drawings()
        
        # Filter out text paths and simple decorative elements
        graphics_paths = [path for path in drawings if not is_text_path(path)]
        
        # Skip if no graphics paths found
        if not graphics_paths:
            logger.info(f"No vector graphics found on page {page_num}")
            return metadata_list
            
        # Create a graph of related paths
        path_relations = {}
        for i, path1 in enumerate(graphics_paths):
            path_relations[i] = set()
            for j, path2 in enumerate(graphics_paths):
                if i != j and paths_are_related(path1, path2):
                    path_relations[i].add(j)
        
        # Find connected components in the graph
        def find_connected_component(start_idx, visited):
            component = {start_idx}
            to_visit = list({start_idx})  # Convert to list to avoid set.pop() issues
            
            while to_visit:
                current = to_visit.pop()  # Now using list.pop()
                for neighbor in path_relations[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        to_visit.append(neighbor)  # Using list.append() instead of set.add()
            
            return component
        
        # Group paths into connected components
        visited = set()
        path_groups = []
        
        for i in range(len(graphics_paths)):
            if i not in visited:
                visited.add(i)
                component = find_connected_component(i, visited)
                
                # Create a group from the component
                group_paths = [graphics_paths[idx] for idx in component]
                
                # Calculate the bounding box for the group
                group_bbox = fitz.Rect()
                for path in group_paths:
                    path_rect = fitz.Rect(path["rect"])
                    group_bbox.include_rect(path_rect)
                
                path_groups.append((group_paths, group_bbox))
        
        # Consolidate graphics using the two-step process
        consolidated_groups = consolidate_graphics(path_groups)
        
        # Process each consolidated group
        for i, (group, bbox) in enumerate(consolidated_groups):
            try:
                # Skip groups with very simple paths
                if len(group) < 2 and len(group[0].get('items', [])) < 3:
                    continue
                    
                # Calculate area and skip if too small, with exception for long, thin graphics
                width = bbox.width
                height = bbox.height
                area = width * height

                is_long_thin_graphic = (
                    (width > MIN_LONG_DIMENSION_FOR_LINES and height < MAX_SHORT_DIMENSION_FOR_LINES) or
                    (height > MIN_LONG_DIMENSION_FOR_LINES and width < MAX_SHORT_DIMENSION_FOR_LINES)
                )

                if not is_long_thin_graphic and area < MIN_VECTOR_AREA:
                    logger.debug(f"Skipping small vector graphic on page {page_num} (size: {width:.1f}x{height:.1f}, area: {area:.1f}) - not long/thin enough or area too small.")
                    continue
                elif is_long_thin_graphic and area < MIN_VECTOR_AREA:
                    logger.debug(f"Keeping long thin graphic vector_{i} on page {page_num} (size: {width:.1f}x{height:.1f}, area: {area:.1f}) despite small area due to aspect ratio.")
                elif not is_long_thin_graphic and area >= MIN_VECTOR_AREA:
                    logger.debug(f"Keeping graphic vector_{i} on page {page_num} (size: {width:.1f}x{height:.1f}, area: {area:.1f}).")

                # Add padding to the bounding box
                padded_bbox = fitz.Rect(
                    bbox.x0 - BORDER_PADDING,
                    bbox.y0 - BORDER_PADDING,
                    bbox.x1 + BORDER_PADDING,
                    bbox.y1 + BORDER_PADDING
                )
                
                # Prepare output
                page_dir = ensure_output_dir(output_dir, pdf_name, page_num)
                image_id = f"vector_{i}"
                output_path = page_dir / f"{image_id}.png"
                
                # Rasterize the padded bounding box area
                pix = page.get_pixmap(clip=padded_bbox, dpi=300)
                pix.save(str(output_path))
                
                # Read the image data for metadata
                with open(output_path, "rb") as f:
                    image_bytes = f.read()
                    
                # Create metadata with padded bbox
                metadata = ImageMetadata.from_image_data(
                    source_pdf_path=pdf_path,
                    page_number=page_num,
                    bbox=(padded_bbox.x0, padded_bbox.y0, padded_bbox.x1, padded_bbox.y1),
                    source_type="vector",
                    original_ext="png",
                    extracted_file_path=output_path,
                    image_data=image_bytes
                )
                
                metadata_list.append(metadata)
                logger.debug(f"Extracted vector graphic {image_id} from page {page_num}")
                
            except Exception as e:
                logger.error(f"Error processing vector group on page {page_num}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path} for vector graphics: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()
            
    logger.info(f"Extracted {len(metadata_list)} vector graphics from page {page_num}")
    return metadata_list

def filter_contained_graphics(metadata_list: List[ImageMetadata]) -> List[ImageMetadata]:
    """
    Filter out graphics that are contained within larger ones.
    
    Args:
        metadata_list: List of ImageMetadata objects to filter
        
    Returns:
        Filtered list of ImageMetadata objects
    """
    sorted_metadata = sorted(
        metadata_list,
        key=lambda m: (m.bbox[2] - m.bbox[0]) * (m.bbox[3] - m.bbox[1]),
        reverse=True
    )
    
    final_filtered_list = []
    contained_indices_set = set()
    
    for i, meta1 in enumerate(sorted_metadata):
        if i in contained_indices_set:
            continue
            
        is_contained_flag = False
        for j in range(len(sorted_metadata)):
            if i == j or j in contained_indices_set: # Don't compare with self or already marked as contained
                continue

            meta2 = sorted_metadata[j]
            # Ensure we are only checking if meta1 is in meta2, where meta2 is potentially larger or processed earlier
            # The sort order implies meta2 could be smaller if j > i, but we check all j != i.
            # The critical part is ensuring meta1 is the one being considered for removal if contained.
            if (meta1.bbox[2]-meta1.bbox[0]) * (meta1.bbox[3]-meta1.bbox[1]) > (meta2.bbox[2]-meta2.bbox[0]) * (meta2.bbox[3]-meta2.bbox[1]):
                # meta1 is larger than meta2, so meta1 cannot be contained in meta2 in a meaningful way for this filter
                # (unless we consider identical bboxes, handled by i==j skip)
                # This optimization assumes we only remove smaller items contained in larger ones.
                # However, the original logic iterated all j != i.
                # To be safe and match original intent of checking against ALL others for containment:
                pass # Allow check, but the sort helps prioritize removing smaller items first. 
                     # Actually, the outer loop processes, and it checks against all *other* items.
                     # The original sorting was to ensure that if A contains B, and B contains C, C is removed first. 
                     # With current loop, if meta1 (larger) is processed, and it contains meta2 (smaller), meta2 should be marked.
                     # The current logic is: meta1 is the candidate. Is it contained in *any other* meta2? 

            if is_contained(meta1.bbox, meta2.bbox): # meta1 is contained in meta2
                is_contained_flag = True
                contained_indices_set.add(i)
                logger.debug(f"Marked graphic {meta1.image_id} (page {meta1.page_number}, area {(meta1.bbox[2]-meta1.bbox[0])*(meta1.bbox[3]-meta1.bbox[1]):.0f}) as contained within {meta2.image_id} (area {(meta2.bbox[2]-meta2.bbox[0])*(meta2.bbox[3]-meta2.bbox[1]):.0f})")
                if meta1.extracted_file_path:
                    # Use a local Path object for unlink without modifying metadata field
                    path_obj = Path(meta1.extracted_file_path) if isinstance(meta1.extracted_file_path, str) else meta1.extracted_file_path
                    if path_obj.exists():
                        try:
                            path_obj.unlink()
                        except OSError as e:
                            logger.error(f"Error unlinking contained file {path_obj}: {e}")
                break # meta1 is contained, no need to check further
                
        if not is_contained_flag:
            final_filtered_list.append(meta1)
            
    # The list is already sorted by area descending if we return final_filtered_list from sorted_metadata.
    # But we are building a new list, so it will be in order of processing non-contained items.
    # For consistency, it might be good to re-sort or return a list that maintains the original relative order of kept items.
    # However, the order of metadata usually doesn't matter for the final JSON output. 
    return final_filtered_list

def process_pdf(pdf_path: Path, output_dir: Path) -> List[ImageMetadata]:
    """
    Main function to process a PDF file and extract all images.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images and metadata
        
    Returns:
        List of all ImageMetadata objects for extracted images
    """
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Get PDF name without extension for output directory
    pdf_name = pdf_path.stem
    all_metadata = []
    
    try:
        # Extract bitmap images
        bitmap_metadata = extract_bitmap_images(pdf_path, output_dir)
        all_metadata.extend(bitmap_metadata)
        
        # Open PDF to get page count
        doc = fitz.open(pdf_path)
        
        # Extract vector graphics from each page
        for page_num in range(len(doc)):
            vector_metadata = extract_vector_graphics(pdf_path, output_dir, page_num)
            all_metadata.extend(vector_metadata)
            
        # Check for edge cases and filter out small/empty images
        check_overlapping_boxes(all_metadata)
        all_metadata = check_small_images(all_metadata)
        
        # Filter out contained graphics
        all_metadata = filter_contained_graphics(all_metadata)
            
        # Create metadata directory
        metadata_dir = output_dir / pdf_name
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata to JSON file
        metadata_path = metadata_dir / "metadata.json"
        
        # Convert metadata to JSON-serializable format
        metadata_json_list = [] # Renamed from metadata_json to avoid confusion
        for metadata_obj in all_metadata: # Iterate over original Pydantic objects
            try:
                # Ensure source_pdf_path and extracted_file_path are strings before model_dump
                if isinstance(metadata_obj.source_pdf_path, Path):
                    metadata_obj.source_pdf_path = str(metadata_obj.source_pdf_path)
                if isinstance(metadata_obj.extracted_file_path, Path):
                    metadata_obj.extracted_file_path = str(metadata_obj.extracted_file_path)

                metadata_dict = metadata_obj.model_dump()

                # Explicitly convert any remaining Path values in the dict to strings
                if isinstance(metadata_dict.get('source_pdf_path'), Path):
                    metadata_dict['source_pdf_path'] = str(metadata_dict['source_pdf_path'])
                if isinstance(metadata_dict.get('extracted_file_path'), Path):
                    metadata_dict['extracted_file_path'] = str(metadata_dict['extracted_file_path'])
                
                metadata_json_list.append(metadata_dict)
            except Exception as e:
                logger.error(f"Error serializing metadata for {metadata_obj.image_id}: {str(e)}")
                # Create a simplified version without the problematic fields, ensuring paths are strings
                simplified = {
                    "image_id": metadata_obj.image_id,
                    "source_pdf_path": str(metadata_obj.source_pdf_path),
                    "page_number": metadata_obj.page_number,
                    "bbox": metadata_obj.bbox,
                    "source_type": metadata_obj.source_type,
                    "original_ext": metadata_obj.original_ext,
                    "extracted_file_path": str(metadata_obj.extracted_file_path) if metadata_obj.extracted_file_path else None,
                    "tags": metadata_obj.tags
                }
                metadata_json_list.append(simplified)
        
        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_json_list, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved metadata for {len(all_metadata)} images to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        # Do not re-raise here if we want the program to continue with other PDFs in a batch
        # For a single PDF run, re-raising is fine. Let's assume re-raise for now.
        raise 
    finally:
        if 'doc' in locals() and doc is not None:
            try:
                doc.close()
            except Exception as e:
                logger.error(f"Error closing PDF document {pdf_path} in finally block: {e}")
            
    return all_metadata 