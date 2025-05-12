import sys
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
import fitz
import random
import re
from tqdm import tqdm

def analyze_text_blocks(page: fitz.Page) -> Dict[str, Any]:
    """
    Analyze text blocks and dictionary structure of a page.

    Parameters
    ----------
    page : fitz.Page
        The page to analyze

    Returns
    -------
    Dict[str, Any]
        Dictionary containing text analysis results
    """
    blocks = page.get_text("blocks")
    dict_structure = page.get_text("dict")
    
    block_samples = []
    if blocks:
        samples = random.sample(blocks, min(3, len(blocks)))
        block_samples = [block[4][:100] + "..." if len(block[4]) > 100 else block[4] for block in samples]
    
    dict_samples = []
    if dict_structure.get("blocks"):
        samples = random.sample(dict_structure["blocks"], min(3, len(dict_structure["blocks"])))
        for block in samples:
            if "lines" in block and block["lines"]:
                text = " ".join(span["text"] for line in block["lines"] 
                              for span in line["spans"] if "text" in span)
                if text:
                    dict_samples.append(text[:100] + "..." if len(text) > 100 else text)
    
    return {
        "block_count": len(blocks),
        "dict_block_count": len(dict_structure.get("blocks", [])),
        "block_samples": block_samples,
        "dict_samples": dict_samples
    }

def get_page_statistics(page: fitz.Page) -> Dict[str, Any]:
    """
    Extract statistics for a single PDF page.

    Parameters
    ----------
    page : fitz.Page
        The PDF page to analyze

    Returns
    -------
    Dict[str, Any]
        Dictionary containing page statistics
    """
    images = page.get_images()
    full_text = page.get_text().strip()
    
    # Get random text snippet if text exists
    text_snippet = ""
    if full_text:
        if len(full_text) <= 100:
            text_snippet = full_text
        else:
            start = random.randint(0, len(full_text) - 100)
            text_snippet = full_text[start:start + 100]
    
    # Get image details
    image_details = []
    for img in images:
        xref = img[0]
        try:
            base_image = page.parent.extract_image(xref)
            image_details.append({
                "width": base_image["width"],
                "height": base_image["height"],
                "type": base_image["ext"]
            })
        except Exception:
            image_details.append({"error": "Could not extract image details"})
    
    text_analysis = analyze_text_blocks(page)
    
    # Extract true vector graphics and discarded graphics
    true_vector_graphics_bboxes, discarded_graphics_bboxes = extract_true_vector_graphics(
        page, return_discarded=True
    )
    
    return {
        "images": len(images),
        "image_details": image_details,
        "vector_drawings": len(true_vector_graphics_bboxes),
        "vector_drawing_bboxes": [list(rect) for rect in true_vector_graphics_bboxes],
        "discarded_vector_drawings": len(discarded_graphics_bboxes),
        "text_blocks": len(page.get_text("blocks")),
        "has_text": bool(full_text),
        "text_snippet": text_snippet,
        "fonts": page.get_fonts(),
        "text_analysis": text_analysis
    }

def adjust_text_bbox_height(rect: fitz.Rect) -> fitz.Rect:
    """
    Adjusts a text bounding box by reducing its height to 1/4 of original and centering it vertically.
    
    Parameters
    ----------
    rect : fitz.Rect
        The original text bounding box
        
    Returns
    -------
    fitz.Rect
        The adjusted bounding box with reduced height
    """
    if not rect.is_valid or rect.is_empty:
        return rect
        
    center_y = (rect.y0 + rect.y1) / 2
    half_height = (rect.y1 - rect.y0) / 8  # Eighth of original height (1/4 of original total)
    
    # Create new rect with same x-coordinates but adjusted y-coordinates
    return fitz.Rect(rect.x0, center_y - half_height, rect.x1, center_y + half_height)

def filter_oversized_boxes(rect: fitz.Rect, page: fitz.Page, max_width: float = 70.0) -> bool:
    """
    Filter out text boxes that exceed a maximum absolute width.

    Parameters
    ----------
    rect : fitz.Rect
        The text bounding box to check
    page : fitz.Page
        The page containing the text box (unused but kept for API compatibility)
    max_width : float, optional
        Maximum absolute width in points, default is 50.0

    Returns
    -------
    bool
        True if the box should be kept, False if it should be filtered out
    """
    if not rect.is_valid or rect.is_empty:
        return False

    return rect.width <= max_width

def get_text_boxes(page: fitz.Page) -> List[fitz.Rect]:
    """
    Get all text bounding boxes from a page using multiple extraction methods.

    Parameters
    ----------
    page : fitz.Page
        The page to analyze

    Returns
    -------
    List[fitz.Rect]
        List of rectangles representing text areas
    """
    boxes = []
    
    # Method 1: Get blocks
    blocks = page.get_text("blocks")
    for block in blocks:
        rect = fitz.Rect(block[:4])
        if filter_oversized_boxes(rect, page):
            boxes.append(adjust_text_bbox_height(rect))
    
    # Method 2: Get dict blocks
    try:
        dict_blocks = page.get_text("dict").get("blocks", [])
        for block in dict_blocks:
            if "bbox" in block:
                rect = fitz.Rect(block["bbox"])
                if filter_oversized_boxes(rect, page):
                    boxes.append(adjust_text_bbox_height(rect))
    except Exception:
        pass
    
    # Method 3: Get words
    try:
        words = page.get_text("words")
        for word in words:
            rect = fitz.Rect(word[:4])
            if filter_oversized_boxes(rect, page):
                boxes.append(adjust_text_bbox_height(rect))
    except Exception:
        pass
    
    # Method 4: Get text with HTML formatting
    try:
        html_dict = page.get_text("html")
        # Extract any rect coordinates from HTML output
        rect_matches = re.findall(r'style="left:(\d+)px; top:(\d+)px; width:(\d+)px; height:(\d+)px', html_dict)
        for match in rect_matches:
            if len(match) == 4:
                left, top, width, height = map(float, match)
                rect = fitz.Rect(left, top, left + width, top + height)
                if filter_oversized_boxes(rect, page):
                    boxes.append(adjust_text_bbox_height(rect))
    except Exception:
        pass
    
    return boxes

def get_inflated_text_bboxes(page: fitz.Page, inflation: float = 5.0) -> List[fitz.Rect]:
    """
    Extracts text block bounding boxes from a page and inflates them.

    Parameters
    ----------
    page : fitz.Page
        The page to analyze.
    inflation : float, optional
        Amount to inflate the bounding boxes by, default is 5.0.

    Returns
    -------
    List[fitz.Rect]
        List of inflated text bounding boxes.
    """
    text_bboxes = []
    blocks = page.get_text("blocks")
    for block in blocks:
        rect = fitz.Rect(block[:4])
        if rect.is_valid and not rect.is_empty and filter_oversized_boxes(rect, page):
            adjusted_rect = adjust_text_bbox_height(rect)
            inflated_rect = adjusted_rect + (-inflation, -inflation, inflation, inflation)
            text_bboxes.append(inflated_rect)
    return text_bboxes

def get_image_bboxes(page: fitz.Page) -> List[fitz.Rect]:
    """
    Extracts bounding boxes of images displayed on a page.

    Parameters
    ----------
    page : fitz.Page
        The page to analyze.

    Returns
    -------
    List[fitz.Rect]
        List of image bounding boxes.
    """
    image_bboxes = []
    image_list = page.get_images(full=False) 
    for img_item in image_list:
        xref = img_item[0]
        rects = page.get_image_rects(xref)
        for r in rects: # Changed variable name from rect to r to avoid conflict
            if isinstance(r, fitz.Rect) and r.is_valid and not r.is_empty:
                image_bboxes.append(r)
    # Remove duplicate rects that might arise if an image is referenced multiple times
    # by converting to a list of tuples, then to a set, then back to a list of Rects.
    if not image_bboxes:
        return []
    unique_rect_coords = sorted(list(set(tuple(r) for r in image_bboxes)))
    return [fitz.Rect(coords) for coords in unique_rect_coords]


def get_initial_vector_shapes(page: fitz.Page) -> List[Dict[str, Any]]:
    """
    Gets all vector drawing primitives from the page.

    Parameters
    ----------
    page : fitz.Page
        The page to analyze.

    Returns
    -------
    List[Dict[str, Any]]
        List of drawing dictionaries (paths).
    """
    return page.get_drawings()


def filter_vector_shapes(
    shapes: List[Dict[str, Any]],
    text_bboxes: List[fitz.Rect],
    image_bboxes: List[fitz.Rect],
    min_segments: int = 1, 
    min_dimension: float = 2.0 
) -> List[fitz.Rect]:
    """
    Filters vector shapes based on overlap with text/images, complexity, and size.

    Parameters
    ----------
    shapes : List[Dict[str, Any]]
        List of raw vector shapes (drawing dictionaries).
    text_bboxes : List[fitz.Rect]
        List of inflated text bounding boxes.
    image_bboxes : List[fitz.Rect]
        List of image bounding boxes.
    min_segments : int, optional
        Minimum number of path drawing commands for a shape, default is 1.
    min_dimension : float, optional
        Minimum width or height for a shape's bounding box, default is 2.0.

    Returns
    -------
    List[fitz.Rect]
        List of bounding boxes for candidate vector graphics.
    """
    candidate_rects = []
    
    # Ensure we're only working with valid text boxes
    valid_text_bboxes = [box for box in text_bboxes if box.is_valid and not box.is_empty]
    
    for shape in shapes:
        shape_rect = fitz.Rect(shape["rect"])
        if not shape_rect.is_valid or shape_rect.is_empty:
            continue

        overlaps_text = any(shape_rect.intersects(text_bbox) for text_bbox in valid_text_bboxes)
        if overlaps_text:
            continue
        
        overlaps_image = any(shape_rect.intersects(image_bbox) for image_bbox in image_bboxes)
        if overlaps_image:
            continue

        num_drawing_commands = len(shape.get("items", []))
        if num_drawing_commands < min_segments:
            continue

        if shape_rect.width < min_dimension or shape_rect.height < min_dimension:
            continue
            
        candidate_rects.append(shape_rect)
    return candidate_rects

def merge_overlapping_rects(rects: List[fitz.Rect], expansion_distance: float = 10.0, max_distance: float = 30.0) -> List[fitz.Rect]:
    """
    Merges overlapping or nearby rectangles into larger bounding boxes.

    Parameters
    ----------
    rects : List[fitz.Rect]
        List of rectangles to merge.
    expansion_distance : float, optional
        Distance to expand rectangles for proximity check, default is 5.0.
    max_distance : float, optional
        Maximum closest-points distance to consider merging, default is 15.0.

    Returns
    -------
    List[fitz.Rect]
        List of merged bounding boxes.
    """
    if not rects:
        return []

    valid_rects = [r for r in rects if r.is_valid and not r.is_empty]
    if not valid_rects:
        return []
    
    # Use progressive merging with increasing distance thresholds
    current_rects = list(valid_rects)
    
    # Multiple passes with increasing distance thresholds
    distance_thresholds = [max_distance, max_distance * 1.5, max_distance * 2.0]
    
    for threshold in distance_thresholds:
        # Stop if we've merged everything into a single rectangle
        if len(current_rects) <= 1:
            break
            
        merged_result = perform_merging_pass(current_rects, expansion_distance, threshold)
        
        # If no further merging occurred, try the next threshold
        if len(merged_result) == len(current_rects):
            continue
        
        current_rects = merged_result
    
    return current_rects

def perform_merging_pass(rects: List[fitz.Rect], expansion_distance: float, max_distance: float) -> List[fitz.Rect]:
    """
    Performs a single pass of rectangle merging.

    Parameters
    ----------
    rects : List[fitz.Rect]
        List of rectangles to merge.
    expansion_distance : float
        Distance to expand rectangles for proximity check.
    max_distance : float
        Maximum distance between rectangles to consider merging.

    Returns
    -------
    List[fitz.Rect]
        List of merged rectangles after one pass.
    """
    current_rects = list(rects)
    
    while True:
        merged_in_pass = False
        next_pass_rects = []
        processed_indices = [False] * len(current_rects)

        for i in range(len(current_rects)):
            if processed_indices[i]:
                continue

            # Start with current_rects[i] as the base for a potential merged rectangle
            merged_rect = current_rects[i]
            processed_indices[i] = True # Mark as processed

            # Its expanded version for checking proximity
            expanded_merged_rect = merged_rect + (-expansion_distance, -expansion_distance, 
                                                  expansion_distance, expansion_distance)

            for j in range(i + 1, len(current_rects)):
                if processed_indices[j]:
                    continue
                
                candidate_rect = current_rects[j]
                
                # Calculate minimum distance between rectangles
                min_distance = get_min_distance_between_rects(merged_rect, candidate_rect)
                
                if min_distance <= max_distance:
                    expanded_candidate_rect = candidate_rect + (-expansion_distance, -expansion_distance,
                                                             expansion_distance, expansion_distance)
                    
                    # Check if expanded rectangles intersect
                    if expanded_merged_rect.intersects(expanded_candidate_rect):
                        merged_rect = merged_rect | candidate_rect # Union of original rects
                        # Update the expanded version of the merged_rect for subsequent checks in this inner loop
                        expanded_merged_rect = merged_rect + (-expansion_distance, -expansion_distance, 
                                                           expansion_distance, expansion_distance)
                        processed_indices[j] = True # Mark as merged into current_merged_rect
                        merged_in_pass = True
            
            next_pass_rects.append(merged_rect) # Add the (potentially grown) merged_rect
        
        current_rects = next_pass_rects
        if not merged_in_pass: # If no merges occurred in a full pass, the process is stable
            break
            
    return current_rects


def prune_final_graphic_rects(rects: List[fitz.Rect], min_dimension: float = 15.0) -> List[fitz.Rect]:
    """
    Prunes merged regions that are too small to be true graphics.

    Parameters
    ----------
    rects : List[fitz.Rect]
        List of merged bounding boxes.
    min_dimension : float, optional
        Minimum width or height for a final graphic, default is 15.0.

    Returns
    -------
    List[fitz.Rect]
        List of finalized graphic bounding boxes.
    """
    final_rects = []
    for rect in rects:
        if rect.is_valid and not rect.is_empty:
            if rect.width >= min_dimension and rect.height >= min_dimension:
                final_rects.append(rect)
    return final_rects


def extract_true_vector_graphics(page: fitz.Page,
                                 text_inflation: float = 5.0,
                                 min_shape_segments: int = 1,
                                 min_shape_dimension: float = 2.0,
                                 cluster_expansion: float = 5.0,
                                 max_merge_distance: float = 20.0,
                                 min_final_graphic_dimension: float = 15.0,
                                 return_discarded: bool = False
                                 ) -> Union[List[fitz.Rect], Tuple[List[fitz.Rect], List[fitz.Rect]]]:
    """
    Extracts bounding boxes of "true" vector graphics from a page.

    Parameters
    ----------
    page : fitz.Page
        The page to analyze.
    text_inflation : float, optional
        Inflation for text bounding boxes. Default is 5.0.
    min_shape_segments : int, optional
        Min path segments for a primitive shape. Default is 1.
    min_shape_dimension : float, optional
        Min dimension for a primitive shape. Default is 2.0.
    cluster_expansion : float, optional
        Expansion for clustering shapes. Default is 5.0.
    max_merge_distance : float, optional
        Maximum distance between graphics to merge. Default is 20.0.
    min_final_graphic_dimension : float, optional
        Min dimension for a final graphic. Default is 15.0.
    return_discarded : bool, optional
        Whether to return discarded vector shapes that overlap with text.

    Returns
    -------
    Union[List[fitz.Rect], Tuple[List[fitz.Rect], List[fitz.Rect]]]
        List of bounding boxes for true vector graphics, and optionally discarded boxes.
    """
    # Use tqdm to track the progress of vector graphic extraction
    with tqdm(total=5, desc="Extracting vector graphics", leave=False) as pbar:
        # Step 1: Get text bounding boxes
        text_bboxes = get_inflated_text_bboxes(page, inflation=text_inflation)
        pbar.update(1)
        
        # Step 2: Get image bounding boxes
        image_bboxes = get_image_bboxes(page)
        pbar.update(1)
        
        # Step 3: Get all vector shapes and process them
        all_vector_shapes = get_initial_vector_shapes(page)
        discarded_text_overlapping_rects = []
        
        if return_discarded:
            # Identify shapes that overlap with text
            valid_text_bboxes = [box for box in text_bboxes if box.is_valid and not box.is_empty]
            
            for shape in all_vector_shapes:
                shape_rect = fitz.Rect(shape["rect"])
                if not shape_rect.is_valid or shape_rect.is_empty:
                    continue
                    
                # Check if the shape meets minimum requirements
                num_drawing_commands = len(shape.get("items", []))
                if num_drawing_commands < min_shape_segments:
                    continue
                    
                if shape_rect.width < min_shape_dimension or shape_rect.height < min_shape_dimension:
                    continue
                    
                # Check if it overlaps with text but not with images
                overlaps_text = any(shape_rect.intersects(text_bbox) for text_bbox in valid_text_bboxes)
                overlaps_image = any(shape_rect.intersects(image_bbox) for image_bbox in image_bboxes)
                
                if overlaps_text and not overlaps_image:
                    discarded_text_overlapping_rects.append(shape_rect)
        pbar.update(1)
        
        # Step 4: Get candidate graphics (non-text-overlapping)
        candidate_graphic_rects = filter_vector_shapes(
            all_vector_shapes, text_bboxes, image_bboxes,
            min_segments=min_shape_segments, min_dimension=min_shape_dimension
        )
        pbar.update(1)
        
        # Step 5: Merge and finalize graphics
        merged_graphic_rects = merge_overlapping_rects(
            candidate_graphic_rects, 
            expansion_distance=cluster_expansion,
            max_distance=max_merge_distance
        )
        
        final_graphic_rects = prune_final_graphic_rects(
            merged_graphic_rects, min_dimension=min_final_graphic_dimension
        )
        pbar.update(1)
    
    if return_discarded:
        return final_graphic_rects, discarded_text_overlapping_rects
    else:
        return final_graphic_rects

def create_visualization(doc: fitz.Document, output_path: Path) -> None:
    """
    Create a visualization PDF highlighting different element types.
    
    Parameters
    ----------
    doc : fitz.Document
        The original PDF document
    output_path : Path
        Path where to save the visualization PDF
    """
    vis_doc = fitz.open()
    
    colors = {
        "text_blocks": (1, 0, 0),       # Red for text blocks
        "vector_graphics": (0, 0, 1),    # Blue for vector graphics
        "original_graphics": (0, 0.7, 0)  # Green for all original vector graphics
    }
    
    for page_num, original_page in enumerate(doc):
        page_width = original_page.rect.width
        page_height = original_page.rect.height
        
        vis_page = vis_doc.new_page(width=page_width, height=page_height)
        vis_page.show_pdf_page(vis_page.rect, doc, page_num)
        
        vis_page.insert_text((50, 40), f"Page {page_num+1} - Combined Elements", 
                         fontsize=12, color=(0, 0, 0))
        
        legend_y = 60
        for element_type, color in colors.items():
            vis_page.draw_rect((50, legend_y, 70, legend_y+10), color=color, fill=color)
            vis_page.insert_text((80, legend_y+7), element_type.replace("_", " ").title(), 
                             fontsize=9, color=(0, 0, 0))
            legend_y += 15
        
        for element_type, color in colors.items():
            element_page = vis_doc.new_page(width=page_width, height=page_height)
            element_page.show_pdf_page(element_page.rect, doc, page_num)
            
            shape = element_page.new_shape()
            shape.draw_rect(element_page.rect)
            shape.finish(color=(1, 1, 1), fill=(1, 1, 1), fill_opacity=0.7)
            shape.commit()
            
            element_page.insert_text(
                (50, 40),
                f"Page {page_num+1} - {element_type.replace('_', ' ').title()} Visualization",
                fontsize=12, color=(0, 0, 0)
            )
            
            shape = element_page.new_shape()
            
            if element_type == "text_blocks":
                # Use filtered text boxes instead of raw blocks
                filtered_boxes = get_text_boxes(original_page)
                
                for adjusted_rect in filtered_boxes:
                    shape.draw_rect(adjusted_rect)
                
                shape.finish(color=color, fill=color, fill_opacity=0.3)
                shape.commit()
                
                # Optional: add a second visualization with words if needed
                blocks = original_page.get_text("blocks")
                filtered_block_count = len([1 for block in blocks if filter_oversized_boxes(fitz.Rect(block[:4]), original_page)])
                
                element_page.insert_text(
                    (50, 60),
                    f"Found {filtered_block_count} text blocks after filtering",
                    fontsize=10, color=(0, 0, 0)
                )
                
            elif element_type == "vector_graphics":
                # Use the new extraction method for true vector graphics
                true_vector_graphics_bboxes = extract_true_vector_graphics(original_page)
                
                for rect in true_vector_graphics_bboxes:
                    shape.draw_rect(rect)
                
                shape.finish(color=color, fill=color, fill_opacity=0.2)
                shape.commit()
                
                element_page.insert_text(
                    (50, 60),
                    f"Found {len(true_vector_graphics_bboxes)} true vector graphic elements",
                    fontsize=10, color=(0, 0, 0)
                )
                
            elif element_type == "original_graphics":
                # Get all original vector shapes
                all_vector_shapes = get_initial_vector_shapes(original_page)
                all_graphic_bboxes = [fitz.Rect(shape["rect"]) for shape in all_vector_shapes 
                                    if "rect" in shape and fitz.Rect(shape["rect"]).is_valid 
                                    and not fitz.Rect(shape["rect"]).is_empty]
                
                for rect in all_graphic_bboxes:
                    shape.draw_rect(rect)
                
                shape.finish(color=color, fill=color, fill_opacity=0.2)
                shape.commit()
                
                element_page.insert_text(
                    (50, 60),
                    f"Found {len(all_graphic_bboxes)} original vector elements (before processing)",
                    fontsize=10, color=(0, 0, 0)
                )
    
    vis_doc.save(output_path)
    vis_doc.close()

def analyze_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Analyze a PDF file and extract detailed statistics.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file

    Returns
    -------
    Dict[str, Any]
        Dictionary containing comprehensive PDF statistics
    """
    doc = fitz.open(pdf_path)
    
    # Create directory for visualizations
    vis_dir = Path.cwd() / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Create visualization PDF
    vis_path = vis_dir / f"visualization_{pdf_path.name}"
    create_visualization(doc, vis_path)
    
    # Get PDF version safely
    try:
        pdf_version = doc.pdf_version
    except AttributeError:
        pdf_version = 0  # Unknown version
    
    stats = {
        "file_size_mb": pdf_path.stat().st_size / (1024 * 1024),
        "page_count": len(doc),
        "metadata": doc.metadata,
        "pdf_version": pdf_version,
        "total_images": 0,
        "total_vector_drawings": 0,
        "pages_with_text": 0,
        "fonts": set(),
        "pages": []
    }
    
    # Process pages without progress bar
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_stats = get_page_statistics(page)
        stats["pages"].append(page_stats)
        
        stats["total_images"] += page_stats["images"]
        stats["total_vector_drawings"] += page_stats["vector_drawings"]
        if page_stats["has_text"]:
            stats["pages_with_text"] += 1
        
        # Collect unique fonts
        for font in page_stats["fonts"]:
            stats["fonts"].add(font[3])
    
    stats["fonts"] = list(stats["fonts"])
    stats["visualization_pdf"] = str(vis_path)
    return stats

def format_stats(stats: Dict[str, Any]) -> str:
    """
    Format the PDF statistics into a readable string.

    Parameters
    ----------
    stats : Dict[str, Any]
        Dictionary containing PDF statistics

    Returns
    -------
    str
        Formatted statistics string
    """
    output = [
        f"PDF Analysis Report",
        f"=================",
        f"File size: {stats['file_size_mb']:.2f} MB",
        f"Total pages: {stats['page_count']}",
        f"PDF version: {stats['pdf_version']}",
        f"Total images: {stats['total_images']}",
        f"Total vector drawings: {stats['total_vector_drawings']}",
        f"Pages containing text: {stats['pages_with_text']}\n",
        f"Metadata:",
        f"---------"
    ]
    
    for key, value in stats["metadata"].items():
        if value:
            output.append(f"{key}: {value}")
    
    if stats["fonts"]:
        output.append("\nFonts used:")
        output.append("-----------")
        for font in stats["fonts"]:
            output.append(f"- {font}")
    
    output.append("\nPage-by-page analysis:")
    output.append("-------------------")
    
    for i, page in enumerate(stats["pages"]):
        output.append(f"Page {i + 1}:")
        output.append(f"  Images: {page['images']}")
        if page["image_details"]:
            for j, img in enumerate(page["image_details"]):
                if "error" in img:
                    output.append(f"    Image {j+1}: {img['error']}")
                else:
                    output.append(f"    Image {j+1}: {img['width']}x{img['height']} {img['type'].upper()}")
        output.append(f"  Vector drawings: {page['vector_drawings']}")
        output.append(f"  Text blocks: {page['text_blocks']}")
        output.append(f"  Contains text: {'Yes' if page['has_text'] else 'No'}")
        
        text_analysis = page["text_analysis"]
        output.append(f"  Text analysis:")
        output.append(f"    Raw blocks: {text_analysis['block_count']}")
        output.append(f"    Dict blocks: {text_analysis['dict_block_count']}")
        
        if text_analysis["block_samples"]:
            output.append("    Block samples:")
            for sample in text_analysis["block_samples"]:
                output.append(f"      - {sample}")
                
        if text_analysis["dict_samples"]:
            output.append("    Dict structure samples:")
            for sample in text_analysis["dict_samples"]:
                output.append(f"      - {sample}")
        
        output.append("")
    
    if "visualization_pdf" in stats:
        output.append("\nVisualization PDF:")
        output.append("-----------------")
        output.append(f"Saved as: {stats['visualization_pdf']}")
        output.append("  The visualization shows different element types with color coding:")
        output.append("  - Red: Text blocks")
        output.append("  - Blue: Vector graphics (filtered)")
        output.append("  - Green: Original vector graphics (before filtering)")
    
    # Add analysis interpretation
    output.append("\nInterpretation of Results:")
    output.append("------------------------")
    
    if stats["total_images"] == 0 and stats["total_vector_drawings"] > 0:
        output.append("- This PDF uses vector graphics rather than raster images")
    
    if stats["pdf_version"] < 1.4:
        output.append("- This PDF uses an older format (pre-1.4) which may affect text extraction")
    
    if stats["pages_with_text"] > 0 and stats["fonts"]:
        if any("Type3" in font for font in stats["fonts"]):
            output.append("- Contains Type3 fonts which often result in poor text extraction")
        if any("TrueType" not in font and "Type1" not in font for font in stats["fonts"]):
            output.append("- Contains non-standard fonts which may affect text extraction quality")
    
    if all("?" in page["text_snippet"] or strange_chars(page["text_snippet"]) > 0.3 for page in stats["pages"] if page["text_snippet"]):
        output.append("- The extracted text appears to contain encoding issues or non-standard characters")
        output.append("  This often happens with custom fonts, scanned PDFs, or PDFs with text as paths")
    
    return "\n".join(output)

def strange_chars(text: str) -> float:
    """
    Calculate the ratio of strange characters in text.
    
    Parameters
    ----------
    text : str
        The text to analyze
        
    Returns
    -------
    float
        Ratio of strange characters
    """
    if not text:
        return 0.0
        
    strange_count = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,;:!?-()[]{}\"'"))
    return strange_count / len(text)

def create_cleaned_pdf(doc: fitz.Document, output_path: Path) -> None:
    """
    Create a new PDF with vector graphics removed.

    Parameters
    ----------
    doc : fitz.Document
        The original PDF document
    output_path : Path
        Path where to save the cleaned PDF
    """
    cleaned_doc = fitz.open()
    
    for page_num, original_page in enumerate(doc):
        # Create new page with same dimensions
        page = cleaned_doc.new_page(
            width=original_page.rect.width,
            height=original_page.rect.height
        )
        
        # Get vector graphics boxes to exclude
        vector_boxes = extract_true_vector_graphics(original_page)
        
        # Create a list of rectangles to redact
        for box in vector_boxes:
            # Create white rectangle to cover vector graphic
            shape = page.new_shape()
            shape.draw_rect(box)
            shape.finish(color=(1, 1, 1), fill=(1, 1, 1))
            shape.commit()
        
        # Show original page content (this will be underneath our white rectangles)
        page.show_pdf_page(page.rect, doc, page_num)
    
    cleaned_doc.save(output_path)
    cleaned_doc.close()

def get_min_distance_between_rects(rect1: fitz.Rect, rect2: fitz.Rect) -> float:
    """
    Calculate the minimum distance between two rectangles.
    
    Parameters
    ----------
    rect1 : fitz.Rect
        First rectangle
    rect2 : fitz.Rect
        Second rectangle
        
    Returns
    -------
    float
        Minimum distance between the rectangles. Returns 0 if they overlap.
    """
    # If rectangles intersect, distance is 0
    if rect1.intersects(rect2):
        return 0.0
    
    # Calculate closest points
    # Horizontal distance
    if rect1.x1 < rect2.x0:  # rect1 is left of rect2
        x_dist = rect2.x0 - rect1.x1
    elif rect2.x1 < rect1.x0:  # rect2 is left of rect1
        x_dist = rect1.x0 - rect2.x1
    else:  # x-overlap
        x_dist = 0
        
    # Vertical distance
    if rect1.y1 < rect2.y0:  # rect1 is above rect2
        y_dist = rect2.y0 - rect1.y1
    elif rect2.y1 < rect1.y0:  # rect2 is above rect1
        y_dist = rect1.y0 - rect2.y1
    else:  # y-overlap
        y_dist = 0
        
    # If one dimension overlaps, return the other dimension's distance
    # If both overlap, distance is 0 (handled by intersects check above)
    if x_dist == 0:
        return y_dist
    if y_dist == 0:
        return x_dist
        
    # Pythagorean theorem for diagonal distance
    return (x_dist**2 + y_dist**2)**0.5

def main():
    """
    Main function to handle command-line operation.
    """
    if len(sys.argv) < 2:
        print("Usage: python inspect_pdf.py <path_to_pdf> [--extract]")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: File '{pdf_path}' does not exist.")
        sys.exit(1)
    
    try:
        # Check if we should extract graphics
        if "--extract" in sys.argv:
            # Create directory for cleaned PDFs
            clean_dir = Path.cwd() / "cleaned_pdfs"
            clean_dir.mkdir(exist_ok=True)
            
            # Create cleaned PDF
            doc = fitz.open(pdf_path)
            clean_path = clean_dir / f"cleaned_{pdf_path.name}"
            create_cleaned_pdf(doc, clean_path)
            doc.close()
            
            print(f"Created cleaned PDF: {clean_path}")
            
        # Always perform analysis
        stats = analyze_pdf(pdf_path)
        print(format_stats(stats))
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
