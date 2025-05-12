## Engineering Brief: PDF Graphics Preprocessing and OCR Integration

**Goal:** Implement a new preprocessing step to deterministically extract graphics (bitmaps, vectors, and residuals) from input PDFs before sending a "stripped" version to the existing Mistral OCR pipeline. Then, re-integrate these extracted graphics into the final Markdown output.

---

### Phase 1: Setup and Configuration

1.  **Create Project Configuration File:**
    *   Create a new YAML file named `config.yaml` in the project root (`/home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/config.yaml`).
    *   Populate it with the following structure. Paths should be relative to the project root or be easily configurable as absolute paths.
        ```yaml
        # filepath: /home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/config.yaml
        # --- General Paths ---
        # Input directory for original PDFs (e.g., where users place PDFs for processing)
        # This might be a temporary staging area or a defined input folder.
        # For now, assume it's a subdirectory within 'originals' or similar.
        # Let's define a base input directory for raw PDFs.
        raw_pdf_input_dir: "originals/raw_pdfs/" # User will place PDFs here
        
        # Base output directory for all processing stages
        processing_output_base_dir: "processing_output/"
        
        # --- Graphics Extraction (New Preprocessing Step) ---
        graphics_extraction:
          # Subdirectory within processing_output_base_dir for this stage's outputs
          output_dir_name: "graphics_pipeline_output/"
        
          # Specific output artifact names/subfolders within graphics_extraction.output_dir_name
          stripped_pdf_subdir: "stripped_pdfs/"      # For PDFs with graphics removed
          graphics_json_filename: "graphics_metadata.json" # Metadata for all extracted graphics
          graphics_assets_subdir: "extracted_graphics/" # For actual image files (png, jpg, svg)
        
          # Settings
          enable_hybrid_residual_processing: true # Master switch for Stage 2 (Residual Raster Sweep)
          raster_dpi: 300                         # DPI for rendering pages in Stage 2
          vector_primitive_text_overlap_threshold: 0.5 # IOU for filtering vector primitives near text
          primitive_clustering_adjacency_gap_pt: 5.0   # Max distance (points) for clustering vector primitives
          min_vector_cluster_area_pt2: 100.0         # Min area (points^2) for a vector cluster to be kept
          residual_stage_min_objects_cutoff: 3       # If < K objects from Stage 1, run Stage 2
          residual_blob_text_overlap_threshold: 0.5  # IOU for filtering residual blobs near text
          residual_clustering_adjacency_gap_pt: 5.0    # Max distance (points) for clustering residual blobs
        
        # --- OCR Processing (Existing Mistral OCR) ---
        ocr_processing:
          # Subdirectory within processing_output_base_dir for this stage's outputs
          # This aligns with the existing OUTPUT_BASE_DIR_NAME in mistral_ocr.py
          output_dir_name: "ocr_output/" # Existing 'ocr_output'
        
          # Specific output artifact names/subfolders within ocr_processing.output_dir_name
          # These should match existing conventions in mistral_ocr.py
          json_output_subdir: "ocr_json/"
          image_output_subdir: "ocr_images/" # This will now store re-integrated graphics
          markdown_output_subdir: "ocr_markdown/"
          log_subdir: "logs/"
        
        # --- Logging ---
        logging:
          # Shared log file for the entire pipeline (preprocessing + OCR)
          # Placed in the root of processing_output_base_dir
          pipeline_log_filename: "pipeline_processing.log"
          level: "INFO" # Default logging level (DEBUG, INFO, WARNING, ERROR)
          
        # --- Final Dataset Output (from md_to_mmmu.py) ---
        dataset_generation:
          # This refers to the existing OUTPUT_BASE_DIR_NAME in md_to_mmmu.py
          output_dir_name: "dataset/"
        ```

2.  **Update Dependencies:**
    *   Add `PyMuPDF` and `opencv-python` to [`requirements.txt`](/home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/requirements.txt). Ensure they are the latest versions.
        ```text
        # filepath: /home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/requirements.txt
        // ...existing code...
        PyMuPDF
        opencv-python
        PyYAML
        ```
    *   Run `pip install -r requirements.txt`.

3.  **Shared Logging Configuration:**
    *   Create a new Python module, e.g., `utils/logger_setup.py`.
    *   This module will contain a function to set up a global logger based on settings in `config.yaml` (e.g., `pipeline_log_filename`, `level`).
    *   Ensure all scripts (`pdf_graphics_extractor.py` (new), [`pdf_ocr/mistral_ocr.py`](/home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/pdf_ocr/mistral_ocr.py), [`process_markdown/md_to_mmmu.py`](/home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/process_markdown/md_to_mmmu.py)) import and use this shared logger setup at the beginning of their execution. Modify existing logging setups in those files.

---

### Phase 2: Graphics Extraction Pipeline (New Script)

Create a new Python script: `pdf_preprocessing/pdf_graphics_extractor.py`. This script will implement Stages 1-4 from the plan.

4.  **Main Orchestration in `pdf_graphics_extractor.py`:**
    *   The script should take an input PDF path and the path to the `config.yaml` file as arguments.
    *   It will manage the creation of output directories specified in `config.yaml` under `processing_output_base_dir/graphics_extraction.output_dir_name/`.
    *   It will call functions for Stage 1, Stage 2 (conditionally), Stage 3, and Stage 4.

5.  **Stage 1 – Object-level Extraction:**
    *   **Page Iteration:**
        *   Implement a function that iterates through each page of the input PDF using PyMuPDF.
    *   **Bitmap XObject Extraction:**
        *   For each page, use `page.get_images(full=True)`.
        *   For each extracted image, save its raw bytes to the configured `graphics_assets_subdir` (e.g., `page{P}_img{xref}.png`). Store its metadata (id, page, bbox, type="bitmap", source_xref, path) in an in-memory list.
    *   **Vector Primitive Harvesting:**
        *   For each page, get the page's display list (`page.get_display_list()`).
        *   Traverse the operators, group consecutive drawing ops into primitives, and compute their bboxes.
        *   Query text boxes using `page.get_text("dict")` or `page.get_text("blocks")`.
        *   Filter out vector primitives whose bbox overlaps significantly (IOU > `vector_primitive_text_overlap_threshold` from config) with any text bbox.
        *   Store remaining primitives' metadata.
    *   **Primitive Clustering:**
        *   Implement a function to build an adjacency graph for the harvested vector primitives (connect if bboxes are within `primitive_clustering_adjacency_gap_pt` or overlap).
        *   Use a Union-Find algorithm to find connected components (clusters).
        *   For each cluster, compute its union bbox and sum of areas. Discard clusters with area < `min_vector_cluster_area_pt2`.
        *   For each valid cluster, extract its drawing commands from the display list and save it as an SVG snippet to `graphics_assets_subdir` (e.g., `page{P}_vec{cluster_id}.svg`).
        *   Record cluster metadata (id, page, bbox, type="vector", path) in the in-memory list.

6.  **Stage 2 – Residual Raster Sweep (Conditional):**
    *   Implement this stage to run only if `enable_hybrid_residual_processing` is true in the config AND if Stage 1 yielded fewer than `residual_stage_min_objects_cutoff` total objects for a given page.
    *   **Prepare "Clean" Render:**
        *   Clone the current page in memory.
        *   For all bboxes of objects extracted in Stage 1 for that page, add redaction annotations to make them invisible (or draw white filled rectangles over them).
        *   Rasterize the redacted page at `raster_dpi` (grayscale) using PyMuPDF.
    *   **Text-Mask Generation:**
        *   On the rasterized image, use PyMuPDF's text extraction (`page.get_text("dict")` on a temporary page created from the raster) to get text bounding boxes in pixel coordinates. Convert these to PDF coordinates.
    *   **CV-based Graphic Detection:**
        *   Binarize the rasterized image (e.g., Otsu's thresholding with OpenCV).
        *   Run connected components analysis (OpenCV's `connectedComponentsWithStats`).
        *   Filter out components whose bboxes have an IOU > `residual_blob_text_overlap_threshold` with any text-mask bbox from the previous step.
        *   Cluster nearby remaining components using the `residual_clustering_adjacency_gap_pt` logic.
        *   For each valid cluster, get its pixel bbox, back-map it to PDF coordinates.
        *   Extract the region corresponding to this cluster from the full-DPI rasterized page (the one *without* redactions from Stage 1 objects, but potentially with Stage 1 text still visible) and save it as a PNG file to `graphics_assets_subdir` (e.g., `page{P}_res{n}.png`).
        *   Record metadata (id, page, bbox, type="residual", path) in the in-memory list.

7.  **Stage 3 – JSON Catalog Assembly:**
    *   Merge all records (from Stage 1 and Stage 2) into a single list.
    *   Sort this list by page number, then by y-coordinate (top-to-bottom).
    *   Implement a deduplication step: if two entries have a very high bbox overlap (e.g., > 95% IOU), keep one (e.g., prefer "bitmap" or "vector" over "residual", or the one with larger area).
    *   Write the final list to the `graphics_json_filename` specified in the config. Include overall metadata in the JSON file (DPI used, thresholds, page dimensions for each page).

8.  **Stage 4 – Produce Graphics-Stripped PDF:**
    *   Load the original input PDF again.
    *   Iterate through each entry in the `graphics_json` catalog.
        *   If `type == "bitmap"`, use `page.delete_image(entry["source_xref"])`.
        *   If `type == "vector"` or `type == "residual"`, add a redaction annotation over `entry["bbox"]` with a white fill and set `apply=True` immediately or apply all at the end.
    *   Save the modified PDF to the configured `stripped_pdf_subdir` with a name like `<original_filename>_stripped.pdf`.

---

### Phase 3: Integration with OCR Pipeline

Modify [`pdf_ocr/mistral_ocr.py`](/home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/pdf_ocr/mistral_ocr.py).

9.  **Update OCR Input:**
    *   Modify the `main` function in [`pdf_ocr/mistral_ocr.py`](/home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/pdf_ocr/mistral_ocr.py) to first call the `pdf_graphics_extractor.py` script (or import its main function) for each input PDF.
    *   The OCR script should now use the `*_stripped.pdf` (output from Stage 4) as its input for the Mistral API.
    *   The OCR script will also need the path to the `graphics_metadata.json` generated by `pdf_graphics_extractor.py` for the re-integration step.

10. **Post-OCR Re-integration (Stage 7):**
    *   In [`pdf_ocr/mistral_ocr.py`](/home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/pdf_ocr/mistral_ocr.py), after receiving the Markdown content from the Mistral API (within the `process_pdf` function or a new function called after it):
        *   Load the corresponding `graphics_metadata.json` file.
        *   The existing `parse_json_to_markdown` function (or a new one) needs to be enhanced. It currently handles images returned by Mistral's OCR. This logic needs to be adapted/replaced.
        *   The goal is to inject placeholders like `<img id="graphic_id_from_json">` or standard Markdown `!Image description` into the OCR-generated Markdown.
        *   To do this:
            *   Parse the OCR's output to understand text flow and bounding boxes of text lines/blocks.
            *   Iterate through the text flow. When the current text position/bbox on a page significantly overlaps (e.g., IOU > 0.3, or geometric proximity) with a bbox of an extracted graphic from `graphics_metadata.json` for that page, inject the placeholder. The placeholder should use the `id` from `graphics_metadata.json` or a relative path to the graphic file.
            *   Ensure the relative paths in the final Markdown point correctly to where the graphics will be stored (see next step).
    *   **Copy Graphics Assets:**
        *   After processing all PDFs, or per PDF, copy all files from the `graphics_assets_subdir` (generated by `pdf_graphics_extractor.py`) into the OCR output's image directory (e.g., ocr_images). This ensures the Markdown image links are valid. The paths in the re-integrated Markdown should be relative to the final Markdown file's location, pointing into this `ocr_images` folder.
    *   The existing `save_images` function in [`pdf_ocr/mistral_ocr.py`](/home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/pdf_ocr/mistral_ocr.py) (which saves base64 images from Mistral) might become redundant or need to be disabled if all graphics are handled by the new pre-processing pipeline. Decide whether to keep it as a fallback or remove it. For now, assume it's disabled if the new pipeline provides graphics.

---

### Phase 4: Testing and Documentation

11. **Testing and Validation (Stage 8):**
    *   **Unit Tests:** Create a `tests/` directory.
        *   Develop minimal PDF files (e.g., one with only a raster image, one with simple vector shapes, one mixed).
        *   Write pytest unit tests for `pdf_graphics_extractor.py` to assert that:
            *   `graphics_metadata.json` contains the expected number and types of objects.
            *   Bboxes are reasonably accurate.
            *   Files are created in `extracted_graphics`.
            *   The `_stripped.pdf` has the graphics removed/redacted.
    *   **Integration Tests:**
        *   Select 2-3 complex Känguru PDF files from your test_pdfs directory.
        *   Run the entire pipeline (graphics extraction -> stripping -> OCR -> re-integration).
        *   Visually inspect the `_stripped.pdf` to confirm graphics removal.
        *   Visually inspect the final Markdown output to ensure image placeholders are correctly positioned relative to the text.
        *   Check that all extracted graphics are copied to the final `ocr_images` folder.
    *   **Metrics (Manual/Semi-Automated):**
        *   For the test PDFs, manually count the number of distinct graphical elements. Compare this with the count from `graphics_metadata.json`.
        *   Compare the text content of Markdown generated from the original PDF (if you have a baseline) vs. Markdown from the `_stripped.pdf`. Recall should be very similar.

12. **Documentation Updates (Stage 9):**
    *   Update [`README.md`](/home/janis/Documents/Studium/Master/Sem2/Guided Research/kaenguru_benchmark/README.md):
        *   Describe the new graphics extraction preprocessing step.
        *   Explain the new configuration options in `config.yaml` related to graphics extraction.
        *   Provide the schema for `graphics_metadata.json`.
        *   Update example CLI invocations if the main pipeline script changes.
        *   Detail the new directory structure under `processing_output_base_dir`.
