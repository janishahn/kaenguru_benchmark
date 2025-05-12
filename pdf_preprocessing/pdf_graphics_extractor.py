#!/usr/bin/env python3
"""
PDF Graphics Extractor

This script extracts graphics (bitmaps, vectors, and residuals) from input PDFs
and produces a "stripped" version for OCR processing. It also generates
a JSON catalog of all extracted graphics for later re-integration.
"""
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import fitz  # PyMuPDF
import cv2
import numpy as np
from copy import deepcopy

# Import shared logger setup
from utils.logger_setup import setup_logging_from_config

logger = logging.getLogger(__name__)

class DisjointSet:
    """
    Union-Find data structure for clustering connected components.

    Attributes
    ----------
    parent : dict
        Stores the parent of each element in a set.
    rank : dict
        Stores the rank (an upper bound on the tree height) for each element.
    """
    def __init__(self) -> None:
        """Initializes empty parent and rank dictionaries."""
        self.parent: Dict[Any, Any] = {}
        self.rank: Dict[Any, int] = {}

    def make_set(self, x: Any) -> None:
        """
        Create a new set containing only the element x.

        Parameters
        ----------
        x : Any
            The element to create a set for.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: Any) -> Any:
        """
        Find the representative (root) of the set containing x.

        Implements path compression.

        Parameters
        ----------
        x : Any
            The element to find.

        Returns
        -------
        Any
            The representative of the set containing x.
        """
        if x not in self.parent:
            self.make_set(x)
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: Any, y: Any) -> None:
        """
        Merge the sets containing x and y.

        Implements union by rank.

        Parameters
        ----------
        x : Any
            An element from the first set.
        y : Any
            An element from the second set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

def calculate_iou(bbox1_coords: Tuple[float, float, float, float],
                  bbox2_coords: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    bbox1_coords : Tuple[float, float, float, float]
        First bounding box coordinates (x0, y0, x1, y1).
    bbox2_coords : Tuple[float, float, float, float]
        Second bounding box coordinates (x0, y0, x1, y1).

    Returns
    -------
    float
        IoU score between 0 and 1.
    """
    bbox1 = fitz.Rect(bbox1_coords)
    bbox2 = fitz.Rect(bbox2_coords)
    
    intersection_rect = bbox1 & bbox2 # Intersection
    
    if intersection_rect.is_empty:
        return 0.0

    intersection_area = intersection_rect.width * intersection_rect.height
    bbox1_area = bbox1.width * bbox1.height
    bbox2_area = bbox2.width * bbox2.height
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area <= 0:
        return 0.0
        
    return intersection_area / union_area

def bbox_distance(bbox1_coords: Tuple[float, float, float, float],
                  bbox2_coords: Tuple[float, float, float, float]) -> float:
    """
    Calculate the minimum Euclidean distance between two bounding boxes.

    Parameters
    ----------
    bbox1_coords : Tuple[float, float, float, float]
        First bounding box coordinates (x0, y0, x1, y1).
    bbox2_coords : Tuple[float, float, float, float]
        Second bounding box coordinates (x0, y0, x1, y1).

    Returns
    -------
    float
        Minimum distance between the boxes (0 if they overlap or touch).
    """
    bbox1 = fitz.Rect(bbox1_coords)
    bbox2 = fitz.Rect(bbox2_coords)

    # Check for overlap
    if not (bbox1 & bbox2).is_empty:
        return 0.0

    # Calculate distance components
    dx = 0.0
    if bbox1.x1 < bbox2.x0: # bbox1 is to the left of bbox2
        dx = bbox2.x0 - bbox1.x1
    elif bbox2.x1 < bbox1.x0: # bbox2 is to the left of bbox1
        dx = bbox1.x0 - bbox2.x1

    dy = 0.0
    if bbox1.y1 < bbox2.y0: # bbox1 is above bbox2
        dy = bbox2.y0 - bbox1.y1
    elif bbox2.y1 < bbox1.y0: # bbox2 is above bbox1
        dy = bbox1.y0 - bbox2.y1
        
    return (dx**2 + dy**2)**0.5


def extract_bitmap_xobjects(page: fitz.Page, page_num: int,
                            graphics_assets_output_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract bitmap images (XObjects) from a PDF page.

    Parameters
    ----------
    page : fitz.Page
        The PDF page to extract images from.
    page_num : int
        The page number (0-based).
    graphics_assets_output_dir : Path
        Output directory for extracted image files.

    Returns
    -------
    List[Dict[str, Any]]
        List of metadata dictionaries for each extracted bitmap image.
    """
    image_list = page.get_images(full=True)
    if not image_list:
        return []

    logger.info(f"Page {page_num + 1}: Found {len(image_list)} raw images.")
    extracted_bitmaps_metadata = []
    doc = page.parent

    for img_info in image_list:
        xref = img_info[0]
        smask_xref = img_info[1]
        
        try:
            base_image_pix = fitz.Pixmap(doc, xref)
            if smask_xref != 0:
                smask_pix = fitz.Pixmap(doc, smask_xref)
                # Ensure smask_pix is grayscale for alpha channel
                if smask_pix.colorspace != fitz.csGRAY:
                    smask_pix = fitz.Pixmap(fitz.csGRAY, smask_pix) # Convert to grayscale
                final_pix = fitz.Pixmap(base_image_pix, smask_pix) # Apply mask
            else:
                final_pix = base_image_pix

            # Convert to PNG bytes
            img_bytes = final_pix.tobytes("png")
            
            # Get image bbox on page
            # Note: get_image_bbox can return multiple bboxes if image is shown multiple times
            # We take the first one for simplicity, or consider all if needed.
            img_bboxes_on_page = page.get_image_bbox(img_info, transform=False) # Get raw bbox
            if not img_bboxes_on_page:
                logger.warning(f"Page {page_num + 1}: Could not find bbox for image xref {xref}.")
                continue # Skip if no bbox found
            
            # For simplicity, use the first bbox. If an image is used multiple times,
            # this might need more sophisticated handling.
            img_bbox = tuple(img_bboxes_on_page[0].irect)  # Use topleft of first instance

            image_id = f"page{page_num + 1}_img{xref}"
            image_filename = f"{image_id}.png"
            image_path = graphics_assets_output_dir / image_filename
            
            with open(image_path, "wb") as img_file:
                img_file.write(img_bytes)

            metadata = {
                "id": image_id,
                "page_num": page_num + 1,
                "bbox": img_bbox, # (x0, y0, x1, y1)
                "type": "bitmap",
                "source_xref": xref,
                "path": str(image_path.relative_to(graphics_assets_output_dir.parent.parent)) # Relative to processing_output_base_dir
            }
            extracted_bitmaps_metadata.append(metadata)
            logger.debug(f"Page {page_num + 1}: Extracted bitmap {image_id}, bbox {img_bbox}")

        except Exception as e:
            logger.error(f"Page {page_num + 1}: Failed to extract image xref {xref}: {e}")
        finally:
            # Clean up Pixmap objects
            if 'final_pix' in locals() and final_pix: final_pix = None
            if 'base_image_pix' in locals() and base_image_pix: base_image_pix = None
            if 'smask_pix' in locals() and smask_pix: smask_pix = None


    return extracted_bitmaps_metadata


def harvest_vector_primitives(page: fitz.Page, page_num: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract vector graphics primitives from a PDF page using page.get_drawings().

    Filters primitives based on overlap with text.

    Parameters
    ----------
    page : fitz.Page
        The PDF page to extract vector primitives from.
    page_num : int
        The page number (0-based).
    config : Dict[str, Any]
        The configuration dictionary.

    Returns
    -------
    List[Dict[str, Any]]
        List of vector primitive metadata dictionaries.
    """
    logger.info(f"Page {page_num + 1}: Harvesting vector primitives.")
    drawings = page.get_drawings()  # Returns a list of dicts, not fitz.Path objects

    text_overlap_threshold = config["graphics_extraction"]["vector_primitive_text_overlap_threshold"]

    # Get text bboxes on the page
    text_bboxes_coords = []
    # text_blocks = page.get_text("dict")["blocks"] # Original line
    text_data = page.get_text("dict") # More robust access
    page_text_blocks = text_data.get("blocks", [])

    for block in page_text_blocks:
        if 'bbox' in block: # Ensure bbox exists
            bbox_coords = block['bbox']
            # Ensure bbox is valid (x0, y0, x1, y1) with x0 <= x1 and y0 <= y1
            if isinstance(bbox_coords, (list, tuple)) and len(bbox_coords) == 4 and \
               bbox_coords[0] <= bbox_coords[2] and bbox_coords[1] <= bbox_coords[3]:
                text_bboxes_coords.append(bbox_coords)
            else:
                logger.warning(f"Page {page_num + 1}: Skipping invalid text block bbox {bbox_coords}")

    harvested_primitives = []
    # primitive_idx = 0 # Original line, not used in the appended dict

    for drawing in drawings:
        if 'rect' not in drawing: # Robustness: check if 'rect' key exists
            logger.warning(f"Page {page_num + 1}: Drawing item missing 'rect' key. Skipping: {drawing.get('type')}")
            continue
            
        drawing_bbox_coords = drawing['rect']
        # Validate drawing_bbox_coords format and values
        if not (isinstance(drawing_bbox_coords, (list, tuple)) and len(drawing_bbox_coords) == 4 and \
                all(isinstance(c, (int, float)) for c in drawing_bbox_coords) and \
                drawing_bbox_coords[0] <= drawing_bbox_coords[2] and \
                drawing_bbox_coords[1] <= drawing_bbox_coords[3]):
            logger.warning(f"Page {page_num + 1}: Skipping invalid or malformed drawing bbox {drawing_bbox_coords}")
            continue

        try:
            primitive_rect = fitz.Rect(drawing_bbox_coords)
        except ValueError as e:
            logger.warning(f"Page {page_num + 1}: Could not create fitz.Rect from drawing bbox {drawing_bbox_coords}. Error: {e}. Skipping.")
            continue
            
        primitive_area = primitive_rect.width * primitive_rect.height

        # Skip zero-area or tiny primitives immediately
        if primitive_area <= 1e-6: # Using a small epsilon for float comparison
            continue

        is_overlapping_with_text = False
        for text_bbox_item_coords in text_bboxes_coords: # text_bbox_item_coords is already validated
            try:
                text_rect = fitz.Rect(text_bbox_item_coords)
            except ValueError as e:
                logger.warning(f"Page {page_num + 1}: Could not create fitz.Rect from text bbox {text_bbox_item_coords}. Error: {e}. Skipping this text_bbox for overlap check.")
                continue

            intersection_rect = primitive_rect & text_rect # Intersection
            
            if not intersection_rect.is_empty:
                intersection_area = intersection_rect.width * intersection_rect.height
                if intersection_area > 1e-6: # Consider only meaningful intersections
                    # Calculate fraction of primitive's area covered by text
                    overlap_fraction_of_primitive = intersection_area / primitive_area
                    
                    if overlap_fraction_of_primitive > text_overlap_threshold:
                        is_overlapping_with_text = True
                        break # This primitive is considered text, no need to check other text boxes
        
        if not is_overlapping_with_text:
            harvested_primitives.append({
                'type': 'vector_primitive',
                'page_num': page_num,
                'bbox': drawing_bbox_coords, # Original coordinate list/tuple
                'drawing': deepcopy(drawing) # The drawing dictionary itself
            })

    logger.info(f"Page {page_num + 1}: Harvested {len(harvested_primitives)} non-text-overlapping vector primitives (using area fraction).")
    return harvested_primitives


def cluster_vector_primitives(primitives: List[Dict[str, Any]], page_num: int, config: Dict[str, Any],
                              graphics_assets_output_dir: Path) -> List[Dict[str, Any]]:
    """
    Cluster harvested vector primitives and save them as SVG assets.

    Parameters
    ----------
    primitives : List[Dict[str, Any]]
        List of harvested vector primitive metadata (must include 'bbox' and 'drawing').
    page_num : int
        The page number (0-based).
    config : Dict[str, Any]
        The configuration dictionary.
    graphics_assets_output_dir : Path
        Output directory for extracted SVG files.

    Returns
    -------
    List[Dict[str, Any]]
        List of metadata dictionaries for each created vector cluster (SVG).
    """
    if not primitives:
        return []

    logger.info(f"Page {page_num + 1}: Clustering {len(primitives)} vector primitives.")

    adjacency_gap_pt = config["graphics_extraction"]["primitive_clustering_adjacency_gap_pt"]
    min_cluster_area_pt2 = config["graphics_extraction"]["min_vector_cluster_area_pt2"]

    ds = DisjointSet()
    for i in range(len(primitives)):
        ds.make_set(i)

    # Build adjacency graph
    for i in range(len(primitives)):
        for j in range(i + 1, len(primitives)):
            bbox1 = primitives[i]["bbox"]
            bbox2 = primitives[j]["bbox"]
            if bbox_distance(bbox1, bbox2) <= adjacency_gap_pt:
                ds.union(i, j)

    clusters_map: Dict[int, List[int]] = {}
    for i in range(len(primitives)):
        root = ds.find(i)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(i)

    clustered_vectors_metadata = []
    cluster_svg_idx = 0
    for root_primitive_idx, primitive_indices in clusters_map.items():
        if not primitive_indices:
            continue
        cluster_primitives = [primitives[i] for i in primitive_indices]
        union_bbox_rect = fitz.Rect()
        for prim_data in cluster_primitives:
            union_bbox_rect.include_rect(fitz.Rect(prim_data["bbox"]))
        if union_bbox_rect.is_empty:
            continue
        cluster_bbox_coords = tuple(union_bbox_rect.irect)
        cluster_area = union_bbox_rect.width * union_bbox_rect.height
        if cluster_area < min_cluster_area_pt2:
            logger.debug(f"Page {page_num + 1}: Discarding small vector cluster (area {cluster_area:.2f} < {min_cluster_area_pt2}).")
            continue
        # Generate SVG for the cluster
        svg_parts = []
        for prim_data in cluster_primitives:
            drawing = prim_data["drawing"]  # This is a dict from get_drawings()
            for item in drawing.get("items", []):
                if item[0] == "path":
                    d_str = item[1]
                    stroke_color = drawing.get("color", "none")
                    fill_color = drawing.get("fill", "none")
                    stroke_width = drawing.get("width", 1.0)
                    style = f'stroke:{stroke_color};fill:{fill_color};stroke-width:{stroke_width}pt;'
                    svg_parts.append(f'<path d="{d_str}" style="{style}"/>' )
        svg_content = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                       f'viewBox="{union_bbox_rect.x0} {union_bbox_rect.y0} {union_bbox_rect.width} {union_bbox_rect.height}">' 
                       f'{"".join(svg_parts)}</svg>')
        cluster_id = f"page{page_num + 1}_vec{cluster_svg_idx}"
        svg_filename = f"{cluster_id}.svg"
        svg_path = graphics_assets_output_dir / svg_filename
        with open(svg_path, "w", encoding="utf-8") as svg_file:
            svg_file.write(svg_content)
        metadata = {
            "id": cluster_id,
            "page_num": page_num + 1,
            "bbox": cluster_bbox_coords,
            "type": "vector",
            "path": str(svg_path.relative_to(graphics_assets_output_dir.parent.parent))
        }
        clustered_vectors_metadata.append(metadata)
        cluster_svg_idx += 1
    logger.info(f"Page {page_num + 1}: Formed {len(clustered_vectors_metadata)} vector graphic clusters.")
    return clustered_vectors_metadata


def process_page_graphics(page: fitz.Page, page_num: int, config: Dict[str, Any],
                          graphics_assets_output_dir: Path) -> List[Dict[str, Any]]:
    """
    Process a single PDF page to extract bitmap, vector, and residual graphics.

    Parameters
    ----------
    page : fitz.Page
        The PDF page object.
    page_num : int
        The page number (0-based).
    config : Dict[str, Any]
        Configuration dictionary.
    graphics_assets_output_dir : Path
        Directory to save extracted graphics assets.

    Returns
    -------
    List[Dict[str, Any]]
        A list of metadata dictionaries for all graphics extracted from this page.
    """
    # Stage 1: Bitmap and vector extraction
    bitmaps = extract_bitmap_xobjects(page, page_num, graphics_assets_output_dir)
    primitives = harvest_vector_primitives(page, page_num, config)
    vectors = cluster_vector_primitives(primitives, page_num, config, graphics_assets_output_dir)
    stage1_objects = bitmaps + vectors

    # Stage 2: Residual raster sweep (conditional)
    residuals = run_residual_raster_sweep(
        page=page,
        page_num=page_num,
        stage1_objects=stage1_objects,
        config=config,
        graphics_assets_output_dir=graphics_assets_output_dir
    )
    return stage1_objects + residuals


def extract_graphics_from_pdf(pdf_path: Path, config_path: Path, project_root: Path) -> Optional[Dict[str, Any]]:
    """
    Main orchestration function to extract graphics from a PDF file.

    Parameters
    ----------
    pdf_path : Path
        Path to the input PDF file.
    config_path : Path
        Path to the YAML configuration file.
    project_root : Path
        The root directory of the project.

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary containing all extracted graphics metadata, or None on failure.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}") # Logger not set up yet
        return None

    # Setup logging (idempotent, safe to call if already configured)
    # setup_logging_from_config(config_path) # This is done in main, before calling this.

    logger.info(f"Starting graphics extraction for PDF: {pdf_path.name}")

    # Define output directories based on config and project_root
    base_output_dir_config = config["processing_output_base_dir"]
    # Ensure base_output_dir is absolute or resolved from project_root
    if Path(base_output_dir_config).is_absolute():
        processing_base_dir = Path(base_output_dir_config)
    else:
        processing_base_dir = project_root / base_output_dir_config
    
    graphics_extraction_config = config["graphics_extraction"]
    graphics_output_dir_name = graphics_extraction_config["output_dir_name"]
    graphics_assets_subdir_name = graphics_extraction_config["graphics_assets_subdir"]
    # stripped_pdf_subdir_name = graphics_extraction_config["stripped_pdf_subdir"] # For Stage 4

    # Specific output paths for this graphics extraction pipeline
    graphics_pipeline_output_dir = processing_base_dir / graphics_output_dir_name
    graphics_assets_output_dir = graphics_pipeline_output_dir / graphics_assets_subdir_name
    # stripped_pdfs_output_dir = graphics_pipeline_output_dir / stripped_pdf_subdir_name # For Stage 4

    # Create output directories
    try:
        graphics_pipeline_output_dir.mkdir(parents=True, exist_ok=True)
        graphics_assets_output_dir.mkdir(parents=True, exist_ok=True)
        # stripped_pdfs_output_dir.mkdir(parents=True, exist_ok=True) # For Stage 4
        logger.info(f"Ensured output directory exists: {graphics_assets_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directories: {e}")
        return None

    all_graphics_metadata: List[Dict[str, Any]] = []
    page_dimensions: Dict[int, Tuple[float, float]] = {}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        return None

    for i, page in enumerate(doc):
        page_metadata = process_page_graphics(page, i, config, graphics_assets_output_dir)
        all_graphics_metadata.extend(page_metadata)
        # Record page dimensions (width, height in points)
        page_dimensions[i + 1] = (page.rect.width, page.rect.height)
    doc.close()
    logger.info(f"Finished graphics extraction for PDF: {pdf_path.name}. Found {len(all_graphics_metadata)} graphics total.")

    # Write graphics_metadata.json (Stage 3)
    graphics_json_path = graphics_pipeline_output_dir / graphics_extraction_config["graphics_json_filename"]
    raster_dpi = graphics_extraction_config.get("raster_dpi", 300)
    assemble_graphics_json_catalog(
        all_graphics_records=all_graphics_metadata,
        output_json_path=graphics_json_path,
        page_dimensions=page_dimensions,
        config=config,
        raster_dpi=raster_dpi
    )

    return {"graphics_metadata": all_graphics_metadata, "config": config}


def run_residual_raster_sweep(
    page: fitz.Page,
    page_num: int,
    stage1_objects: List[Dict[str, Any]],
    config: Dict[str, Any],
    graphics_assets_output_dir: Path,
    full_dpi_raster: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Perform Stage 2: Residual Raster Sweep for a PDF page.

    Parameters
    ----------
    page : fitz.Page
        The PDF page object.
    page_num : int
        The page number (0-based).
    stage1_objects : List[Dict[str, Any]]
        List of objects extracted in Stage 1 for this page.
    config : Dict[str, Any]
        Configuration dictionary.
    graphics_assets_output_dir : Path
        Directory to save extracted graphics assets.
    full_dpi_raster : Optional[np.ndarray]
        Full-DPI rasterized page (without redactions), if available.

    Returns
    -------
    List[Dict[str, Any]]
        List of metadata dictionaries for each extracted residual graphic.
    """
    # Check if Stage 2 should run
    enable_hybrid = config["graphics_extraction"].get("enable_hybrid_residual_processing", False)
    min_cutoff = config["graphics_extraction"].get("residual_stage_min_objects_cutoff", 3)
    if not enable_hybrid or len(stage1_objects) >= min_cutoff:
        logger.info(f"Page {page_num + 1}: Skipping residual raster sweep (Stage 2).")
        return []

    logger.info(f"Page {page_num + 1}: Running residual raster sweep (Stage 2).")
    raster_dpi = config["graphics_extraction"].get("raster_dpi", 300)
    blob_text_overlap_thresh = config["graphics_extraction"].get("residual_blob_text_overlap_threshold", 0.5)
    cluster_gap_pt = config["graphics_extraction"].get("residual_clustering_adjacency_gap_pt", 5.0)

    # 1. Prepare "clean" render: clone page, redact Stage 1 bboxes, rasterize
    doc = page.parent
    temp_doc = fitz.open()
    temp_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    temp_page = temp_doc[0]
    for obj in stage1_objects:
        bbox = obj["bbox"]
        temp_page.add_redact_annot(fitz.Rect(bbox), fill=(1, 1, 1))
    temp_page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    pix = temp_page.get_pixmap(dpi=raster_dpi, colorspace=fitz.csGRAY)
    raster_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    # 2. Text-mask generation: get text bboxes in pixel coords, convert to PDF coords
    text_blocks = temp_page.get_text("dict")["blocks"]
    text_bboxes_pdf = [tuple(block["bbox"]) for block in text_blocks if block["type"] == 0]
    # Convert PDF bboxes to pixel bboxes
    def pdf_to_pixel(bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox
        w, h = temp_page.rect.width, temp_page.rect.height
        px = lambda x, axis: int(round(x * (pix.width if axis == 'x' else pix.height) / (w if axis == 'x' else h)))
        return (px(x0, 'x'), px(y0, 'y'), px(x1, 'x'), px(y1, 'y'))
    text_bboxes_px = [pdf_to_pixel(b) for b in text_bboxes_pdf]

    # 3. CV-based graphic detection
    # Binarize
    _, bin_img = cv2.threshold(raster_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - bin_img, connectivity=8)
    # Each component: stats[i] = [x, y, w, h, area]
    residual_candidates = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        comp_bbox_px = (x, y, x + w, y + h)
        # Filter out if IOU > threshold with any text bbox
        overlaps_text = False
        for tb in text_bboxes_px:
            # Compute IOU in pixel space
            xi0 = max(comp_bbox_px[0], tb[0])
            yi0 = max(comp_bbox_px[1], tb[1])
            xi1 = min(comp_bbox_px[2], tb[2])
            yi1 = min(comp_bbox_px[3], tb[3])
            inter_area = max(0, xi1 - xi0) * max(0, yi1 - yi0)
            comp_area = (comp_bbox_px[2] - comp_bbox_px[0]) * (comp_bbox_px[3] - comp_bbox_px[1])
            tb_area = (tb[2] - tb[0]) * (tb[3] - tb[1])
            union_area = comp_area + tb_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            if iou > blob_text_overlap_thresh:
                overlaps_text = True
                break
        if not overlaps_text:
            residual_candidates.append({"label": i, "bbox_px": comp_bbox_px, "area": area})

    # 4. Cluster nearby components
    ds = DisjointSet()
    for idx in range(len(residual_candidates)):
        ds.make_set(idx)
    for i in range(len(residual_candidates)):
        for j in range(i + 1, len(residual_candidates)):
            b1 = residual_candidates[i]["bbox_px"]
            b2 = residual_candidates[j]["bbox_px"]
            # Distance in pixels (use cluster_gap_pt scaled to pixels)
            gap_px = int(round(cluster_gap_pt * pix.width / temp_page.rect.width))
            # If bboxes are close or overlap, cluster
            dx = max(b1[0] - b2[2], b2[0] - b1[2], 0)
            dy = max(b1[1] - b2[3], b2[1] - b1[3], 0)
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist <= gap_px:
                ds.union(i, j)
    # Group by cluster
    clusters: Dict[int, List[int]] = {}
    for idx in range(len(residual_candidates)):
        root = ds.find(idx)
        clusters.setdefault(root, []).append(idx)

    # 5. For each valid cluster, extract region from full-DPI raster, save PNG, record metadata
    residuals_metadata = []
    cluster_idx = 0
    # If full_dpi_raster is not provided, rasterize original page
    if full_dpi_raster is None:
        orig_pix = page.get_pixmap(dpi=raster_dpi, colorspace=fitz.csGRAY)
        full_dpi_raster = np.frombuffer(orig_pix.samples, dtype=np.uint8).reshape(orig_pix.height, orig_pix.width)
    for cluster_indices in clusters.values():
        # Union bbox in pixel space
        x0 = min(residual_candidates[i]["bbox_px"][0] for i in cluster_indices)
        y0 = min(residual_candidates[i]["bbox_px"][1] for i in cluster_indices)
        x1 = max(residual_candidates[i]["bbox_px"][2] for i in cluster_indices)
        y1 = max(residual_candidates[i]["bbox_px"][3] for i in cluster_indices)
        # Back-map to PDF coordinates
        def px_to_pdf(x: int, y: int) -> Tuple[float, float]:
            w, h = temp_page.rect.width, temp_page.rect.height
            return (x * w / pix.width, y * h / pix.height)
        pdf_x0, pdf_y0 = px_to_pdf(x0, y0)
        pdf_x1, pdf_y1 = px_to_pdf(x1, y1)
        pdf_bbox = (pdf_x0, pdf_y0, pdf_x1, pdf_y1)
        # Extract region from full_dpi_raster
        region = full_dpi_raster[y0:y1, x0:x1]
        asset_id = f"page{page_num + 1}_res{cluster_idx}"
        asset_filename = f"{asset_id}.png"
        asset_path = graphics_assets_output_dir / asset_filename
        cv2.imwrite(str(asset_path), region)
        metadata = {
            "id": asset_id,
            "page_num": page_num + 1,
            "bbox": pdf_bbox,
            "type": "residual",
            "path": str(asset_path.relative_to(graphics_assets_output_dir.parent.parent))
        }
        residuals_metadata.append(metadata)
        logger.debug(f"Page {page_num + 1}: Saved residual cluster {asset_id}, bbox {pdf_bbox}")
        cluster_idx += 1
    logger.info(f"Page {page_num + 1}: Extracted {len(residuals_metadata)} residual graphics.")
    return residuals_metadata


def assemble_graphics_json_catalog(
    all_graphics_records: List[Dict[str, Any]],
    output_json_path: Path,
    page_dimensions: Dict[int, Tuple[float, float]],
    config: Dict[str, Any],
    raster_dpi: int
) -> None:
    """
    Assemble and write the graphics metadata JSON catalog for all extracted graphics.

    Parameters
    ----------
    all_graphics_records : List[Dict[str, Any]]
        List of all graphics metadata records (from Stage 1 and Stage 2).
    output_json_path : Path
        Path to write the JSON catalog file.
    page_dimensions : Dict[int, Tuple[float, float]]
        Mapping from page number to (width, height) in PDF points.
    config : Dict[str, Any]
        The configuration dictionary.
    raster_dpi : int
        The DPI used for rasterization (for metadata).
    """
    # Sort by page number, then by y0 (top-to-bottom)
    sorted_records = sorted(
        all_graphics_records,
        key=lambda r: (r["page_num"], r["bbox"][1])
    )

    # Deduplicate: if two entries have IOU > 0.95, keep one (prefer bitmap/vector over residual, or larger area)
    def record_area(rec):
        x0, y0, x1, y1 = rec["bbox"]
        return abs((x1 - x0) * (y1 - y0))

    deduped = []
    for rec in sorted_records:
        keep = True
        for other in deduped:
            if rec["page_num"] == other["page_num"]:
                iou = calculate_iou(tuple(rec["bbox"]), tuple(other["bbox"]))
                if iou > 0.95:
                    # Prefer bitmap/vector over residual, or larger area
                    if rec["type"] == "residual" and other["type"] in ("bitmap", "vector"):
                        keep = False
                        break
                    elif other["type"] == "residual" and rec["type"] in ("bitmap", "vector"):
                        deduped.remove(other)
                        break
                    else:
                        # Keep the one with larger area
                        if record_area(rec) > record_area(other):
                            deduped.remove(other)
                            break
                        else:
                            keep = False
                            break
        if keep:
            deduped.append(rec)

    # Build metadata
    catalog = {
        "graphics": deduped,
        "metadata": {
            "raster_dpi": raster_dpi,
            "vector_primitive_text_overlap_threshold": config["graphics_extraction"]["vector_primitive_text_overlap_threshold"],
            "primitive_clustering_adjacency_gap_pt": config["graphics_extraction"]["primitive_clustering_adjacency_gap_pt"],
            "min_vector_cluster_area_pt2": config["graphics_extraction"]["min_vector_cluster_area_pt2"],
            "residual_stage_min_objects_cutoff": config["graphics_extraction"]["residual_stage_min_objects_cutoff"],
            "residual_blob_text_overlap_threshold": config["graphics_extraction"]["residual_blob_text_overlap_threshold"],
            "residual_clustering_adjacency_gap_pt": config["graphics_extraction"]["residual_clustering_adjacency_gap_pt"],
            "page_dimensions": page_dimensions
        }
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote graphics metadata JSON catalog to {output_json_path}")


def produce_graphics_stripped_pdf(
    input_pdf_path: Path,
    graphics_json_path: Path,
    output_pdf_path: Path
) -> None:
    """
    Produce a graphics-stripped PDF by removing or redacting graphics as specified in the graphics catalog.

    Parameters
    ----------
    input_pdf_path : Path
        Path to the original input PDF.
    graphics_json_path : Path
        Path to the graphics metadata JSON file.
    output_pdf_path : Path
        Path to save the stripped PDF.
    """
    import fitz
    import json

    doc = fitz.open(str(input_pdf_path))
    with open(graphics_json_path, 'r', encoding='utf-8') as f:
        graphics_catalog = json.load(f)
    graphics_list = graphics_catalog.get('graphics', graphics_catalog)  # fallback if not wrapped

    for entry in graphics_list:
        page_num = entry['page']
        bbox = entry['bbox']
        gtype = entry['type']
        page = doc[page_num]
        if gtype == 'bitmap':
            xref = entry.get('source_xref')
            if xref is not None:
                try:
                    page.delete_image(xref)
                except Exception as e:
                    logger.warning(f"Failed to delete image xref {xref} on page {page_num+1}: {e}")
        elif gtype in ('vector', 'residual'):
            try:
                rect = fitz.Rect(bbox)
                annot = page.add_redact_annot(rect, fill=(1, 1, 1))
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            except Exception as e:
                logger.warning(f"Failed to redact bbox {bbox} on page {page_num+1}: {e}")

    doc.save(str(output_pdf_path), garbage=4, deflate=True)
    logger.info(f"Saved graphics-stripped PDF to {output_pdf_path}")


def main(input_pdf_path: Path, config_path: Path) -> None:
    """
    Main entry point for programmatic use: extract graphics and produce stripped PDF for a given PDF and config.

    Parameters
    ----------
    input_pdf_path : Path
        Path to the input PDF file.
    config_path : Path
        Path to the project's config.yaml file.
    Returns
    -------
    None
    """
    if not input_pdf_path.is_file():
        print(f"Error: Input PDF not found at {input_pdf_path}")
        return
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        return
    project_root = config_path.parent
    try:
        setup_logging_from_config(config_path)
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger.error(f"Logging setup failed: {e}. Using basic console logging.")
    logger.info(f"Project root identified as: {project_root}")
    extraction_results = extract_graphics_from_pdf(input_pdf_path, config_path, project_root)
    if extraction_results:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        graphics_extraction_cfg = config["graphics_extraction"]
        processing_output_base_dir = config["processing_output_base_dir"]
        if Path(processing_output_base_dir).is_absolute():
            processing_base_dir = Path(processing_output_base_dir)
        else:
            processing_base_dir = project_root / processing_output_base_dir
        graphics_output_dir = processing_base_dir / graphics_extraction_cfg["output_dir_name"]
        stripped_pdf_subdir = graphics_output_dir / graphics_extraction_cfg["stripped_pdf_subdir"]
        stripped_pdf_subdir.mkdir(parents=True, exist_ok=True)
        graphics_json_path = graphics_output_dir / graphics_extraction_cfg["graphics_json_filename"]
        output_pdf_path = stripped_pdf_subdir / f"{input_pdf_path.stem}_stripped.pdf"
        produce_graphics_stripped_pdf(input_pdf_path, graphics_json_path, output_pdf_path)
        logger.info(f"Graphics-stripped PDF written to {output_pdf_path}")
        logger.info("Graphics extraction process completed successfully.")
    else:
        logger.error("Graphics extraction process failed.")

def main_cli() -> None:
    """
    Command-line interface for the PDF graphics extractor.
    """
    parser = argparse.ArgumentParser(description="Extracts graphics from PDF files based on a configuration.")
    parser.add_argument("input_pdf", type=str, help="Path to the input PDF file.")
    parser.add_argument("config_file", type=str, help="Path to the project's config.yaml file.")
    args = parser.parse_args()
    input_pdf_path = Path(args.input_pdf).resolve()
    config_path = Path(args.config_file).resolve()
    main(input_pdf_path, config_path)

if __name__ == "__main__":
    main_cli()