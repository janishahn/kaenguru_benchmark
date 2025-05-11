**Objective**: To refactor and enhance the existing PDF-to-Markdown conversion pipeline by improving image extraction, introducing a pre-OCR image stripping step, refining Markdown assembly, and adding image post-processing capabilities.

---

### Phase 1: Core Extraction Module Enhancement (Corresponds to Plan Sections 2 & 3)

**Goal**: Develop a robust image extraction module that can handle various image types and store them with detailed metadata.

**Step 1.1: Setup New Extraction Sub-module**
    *   **Task**: Create a new directory `src/pdf_extract/` (or similar, e.g., `pdf_processing/extract/` if `src` doesn't fit the project structure).
    *   **Libraries**: Add `PyMuPDF` (`fitz`) and `poppler-utils` (if `pdfimages` is chosen) to `requirements.txt`.
        *   *Note*: `pdfimages` is a command-line tool. Interaction might be via `subprocess`. PyMuPDF is a Python library.
    *   **Files**:
        *   `src/pdf_extract/image_extractor.py`: Main logic for image extraction.
        *   `src/pdf_extract/metadata.py`: Define Pydantic models for image metadata.

**Step 1.2: Define Image Metadata Schema** (Plan 2.2)
    *   **Task**: In `src/pdf_extract/metadata.py`, define a Pydantic model (e.g., `ImageMetadata`) with fields:
        *   `source_pdf_path: str`
        *   `page_number: int`
        *   `bbox: Tuple[float, float, float, float]` (x0, y0, x1, y1)
        *   `image_id: str` (e.g., UUID or a hash of image data)
        *   `source_type: Literal["bitmap", "vector", "raw_embedded"]`
        *   `original_ext: str` (e.g., "jpg", "png", "svg", "pdf")
        *   `extracted_file_path: str` (path where the extracted image is saved)
        *   `raw_image_data_hash: Optional[str]` (SHA256 hash of the raw image bytes, for deduplication or integrity checks)

**Step 1.3: Implement Bitmap Image Extraction** (Plan 2.1, 3.2)
    *   **Task**: In `src/pdf_extract/image_extractor.py`, implement a function `extract_bitmap_images(pdf_path: Path) -> List[ImageMetadata]`.
    *   **Engine**: Use PyMuPDF (`fitz`) to iterate through pages and extract bitmap images (e.g., `page.get_images(full=True)`). Poppler's `pdfimages` is an alternative. PyMuPDF is likely easier to integrate for metadata like bounding boxes.
    *   **Process**:
        1.  For each image, retrieve its raw bytes and original extension.
        2.  Calculate its bounding box on the page.
        3.  Generate a unique ID.
        4.  Populate the `ImageMetadata` model.
        5.  Save the raw image bytes to a structured path (e.g., `output/extracted_images/{pdf_name}/page_{n}/img_{id}.{ext}`). The path should be stored in `extracted_file_path`.
        *   **No Re-encoding**: Save the original image stream bytes.

**Step 1.4: Implement Vector Image Extraction (Initial)** (Plan 2.1, 3.3)
    *   **Task**: In `src/pdf_extract/image_extractor.py`, implement `extract_vector_graphics(pdf_path: Path, page_num: int) -> List[ImageMetadata]`.
    *   **Engine**: Use PyMuPDF (`fitz`) to get drawing objects (`page.get_drawings()`).
    *   **Process**:
        1.  Identify vector graphics (paths, shapes). This can be complex as "drawings" can include text rendering paths. Focus on non-text paths first.
        2.  For each identified vector graphic (or group of paths forming a distinct element):
            *   Determine its bounding box.
            *   Render it to a separate file (e.g., PNG or SVG using PyMuPDF's capabilities like `Pixmap(doc, drawing_rect)` or by creating a new PDF page with just that drawing and then converting). SVGs are preferable for scalability if the rendering engine supports it well.
            *   Populate `ImageMetadata` (source_type: "vector").
            *   Save to the structured path.

**Step 1.5: Metadata Storage** (Plan 2.3)
    *   **Task**:
        1.  Create a main function in `image_extractor.py` that takes a PDF path, calls both bitmap and vector extraction.
        2.  Collect all `ImageMetadata` objects.
        3.  Store this list as a JSON file (e.g., `output/extracted_images/{pdf_name}/metadata.json`).

**Step 1.6: Edge Case Flagging (Basic)** (Plan 3.4)
    *   **Task**: In `image_extractor.py`, after collecting all metadata:
        *   Implement a function to check for overlapping bounding boxes (e.g., using Intersection over Union - IoU). Log warnings if significant overlap is detected.
        *   Implement a function to check image area (width * height from bbox). If below a configurable threshold (e.g., 50x50 pixels for bitmaps), add a tag like "small_artifact" or similar to the `ImageMetadata` (requires adding an optional `tags: List[str]` field to the model).

**Step 1.7: Initial Integration & QA** (Plan 3.5)
    *   **Task**: Modify `pdf_ocr/mistral_ocr.py` (or create a new main script).
        1.  Before the current OCR processing, call the new image extraction module.
        2.  For a few sample PDFs, manually verify:
            *   The number of extracted images.
            *   The content of a few extracted bitmaps (pixel-perfect if possible).
            *   The content of a few extracted vector graphics.
            *   The correctness of `metadata.json`.
    *   **Output**: A directory `output/extracted_images/{pdf_name}/` containing images and `metadata.json`.

---

### Phase 2: Create Text-Only PDF for OCR (Corresponds to Plan Section 4)

**Goal**: Generate a version of the PDF with all images removed to optimize the OCR stage.

**Step 2.1: Implement Image Stripping Function** (Plan 4.1)
    *   **Task**: Create a new script/module, e.g., `src/pdf_tools/pdf_modifier.py`.
    *   Implement `create_text_only_pdf(original_pdf_path: Path, output_pdf_path: Path, images_to_remove: List[ImageMetadata])`.
    *   **Method**:
        *   Use PyMuPDF. Load the `original_pdf_path`.
        *   For each page, identify and delete the images. PyMuPDF's `page.delete_image(xref)` or redaction annotations can be used. The `ImageMetadata` (specifically `bbox` and `page_number`) will be crucial for identifying which image elements to remove accurately. If image XREF (cross-reference ID) is stored in metadata during extraction, it simplifies deletion.
        *   Save the modified PDF to `output_pdf_path`.
        *   **Decision**: Decide whether to remove vector graphics too. The plan suggests "or remove vectors too if OCR should only see glyphs." Initially, try removing only bitmaps identified in Phase 1.

**Step 2.2: Integrate Stripping into the Main Pipeline**
    *   **Task**: In the main script (`mistral_ocr.py` or new):
        1.  After Phase 1 (image extraction and metadata generation).
        2.  Call `create_text_only_pdf`, using the `metadata.json` to guide which images to strip.
        3.  The output is a new PDF, e.g., `output/text_only_pdfs/{pdf_name}_text_only.pdf`.

**Step 2.3: Validate Text Integrity** (Plan 4.2)
    *   **Task**: For a few sample PDFs:
        1.  Manually open the original PDF and the `_text_only.pdf`.
        2.  Visually compare several pages to ensure no text has been inadvertently lost or corrupted during the image stripping process.

---

### Phase 3: Integrate with OCR Stage (Corresponds to Plan Section 5)

**Goal**: Feed the text-only PDF to the OCR engine and verify improvements.

**Step 3.1: Modify OCR Input** (Plan 5.1)
    *   **Task**: In `pdf_ocr/mistral_ocr.py`:
        *   Change the `process_pdf` function (or its equivalent) to take the path of the `_text_only.pdf` generated in Phase 2 as input for the Mistral OCR API call.
        *   The existing image saving logic in `save_images` within `mistral_ocr.py` (which saves images returned by Mistral) might become redundant or needs careful review. The plan implies we use *our* extracted images. Mistral might still return its own image data/placeholders. We need to decide if we disable Mistral's image output (`include_image_base64=False`?) and rely solely on our pre-extracted images.
        *   **Crucial Decision**: The current `parse_json_to_markdown` uses image paths from Mistral's response. This will need to be heavily modified to use our `metadata.json` from Phase 1.

**Step 3.2: Verify Performance** (Plan 5.2)
    *   **Task**:
        1.  Process a few large PDFs with the original pipeline (OCR on full PDF). Record token usage (if available from API response/logs) and processing time.
        2.  Process the same PDFs with the new pipeline (OCR on text-only PDF). Record token usage and time.
        3.  Compare results. Expect fewer tokens and faster OCR processing.

---

### Phase 4: Design and Implement Markdown Assembly (Corresponds to Plan Section 6)

**Goal**: Reconstruct the document in Markdown, correctly interleaving text blocks from OCR and images extracted in Phase 1.

**Step 4.1: Adapt Markdown Generation Logic** (Plan 6.1)
    *   **Task**: Refactor or replace `parse_json_to_markdown` in `pdf_ocr/mistral_ocr.py`.
        *   **Inputs**:
            *   OCR output (text blocks with their bounding boxes, from Mistral's JSON response on the *text-only PDF*).
            *   The `metadata.json` (list of `ImageMetadata` from our Phase 1 extraction).
        *   **Algorithm**:
            1.  For each page:
                a.  Get all text blocks from OCR for that page. Sort them primarily by top-coordinate (y1 or y0), then secondarily by left-coordinate (x0).
                b.  Get all images from our `metadata.json` for that page. Sort them by their top-coordinate (y0).
                c.  Iterate through the sorted text blocks and sorted images.
                d.  Maintain pointers to the current text block and current image.
                e.  If the current image's `bbox.y0` is less than or equal to the current text block's `bbox.y0` (or some other heuristic to decide it comes "before"), emit the Markdown for the image: `![](path/to/our/extracted_image.ext)`. Advance the image pointer.
                f.  Else, emit the text from the text block. Advance the text block pointer.
                g. Handle cases where multiple images might appear before the next text block, or multiple text blocks before the next image.
                h. This simple Y-sort might need refinement for complex layouts (e.g., multi-column text, images embedded within a paragraph). Consider a more sophisticated layout analysis if needed, but start simple.

**Step 4.2: Image Placeholder Numbering** (Plan 6.2)
    *   **Task**:
        *   Ensure image paths used in Markdown are relative to the final Markdown file location (e.g., `../extracted_images/{pdf_name}/page_{n}/img_{id}.{ext}`).
        *   Decide on image captioning/alt-text: `![Image {global_id}]` or `![Page {page_num} Image {page_local_id}]`. The path itself is unique. The alt text can be simpler. Using `image_id` from metadata could be an option: `![Image {metadata.image_id}]`.
        *   Ensure deterministic output: sorting must be stable.

**Step 4.3: Dry Run and Verification** (Plan 6.3)
    *   **Task**:
        1.  For one or two complex PDFs, run the new assembly logic.
        2.  Instead of writing to a file, log the sequence of (text chunk / image placeholder) being generated.
        3.  Manually compare this sequence against the original PDF layout to verify correctness.

---

### Phase 5: Image Post-Processing & Heuristics (Corresponds to Plan Section 7)

**Goal**: Clean up extracted images and filter unwanted artifacts. This phase can run after initial image extraction (Phase 1) and before Markdown assembly (Phase 4), or as a separate post-processing step on the `output/extracted_images` directory.

**Step 5.1: Implement Tiny Artifact Filter** (Plan 7.1)
    *   **Task**: Create a script/module, e.g., `src/image_processing/filters.py`.
    *   Implement `filter_small_images(metadata_list: List[ImageMetadata], min_pixel_area: int, image_base_dir: Path) -> List[ImageMetadata]`.
    *   **Method**:
        *   Iterate through `ImageMetadata`. For bitmap images, load them using a library like Pillow.
        *   Calculate pixel area (width * height).
        *   If below `min_pixel_area`, mark them for removal (e.g., add a tag "filtered_small" to metadata, or create a new list of metadata to keep).
        *   Optionally, delete the image file. (Careful: ensure this is desired behavior).

**Step 5.2: Split Oversized Strips (Optional - Advanced)** (Plan 7.2)
    *   **Task**: This is more complex. Initially, focus on other steps. If implemented:
        *   Detect images with extreme aspect ratios (e.g., >4:1 or <1:4).
        *   Use whitespace detection (e.g., profile projection) or other CV techniques to find split points.
        *   Update metadata: one image becomes multiple, with new bboxes and IDs.

**Step 5.3: Ignore Unwanted Logos/Headers (Blocklist)** (Plan 7.3)
    *   **Task**: In `src/image_processing/filters.py`:
    *   Implement `filter_blocklisted_images(metadata_list: List[ImageMetadata], blocklist_config: Dict) -> List[ImageMetadata]`.
    *   **Blocklist Config**: Could be a JSON/YAML file defining:
        *   Filename patterns (e.g., `*logo*.png`).
        *   Bounding box zones on specific pages to ignore (e.g., page 1, top 10% height for headers).
    *   **Method**: Iterate through metadata, check against blocklist rules, and mark/filter.

**Step 5.4: Integrate Post-Processing**
    *   **Task**:
        1.  Decide where to run these filters. A good place is after Phase 1 (raw extraction) and before Phase 2 (PDF stripping) or Phase 4 (Markdown assembly). The `metadata.json` would be updated.
        2.  The Markdown assembly (Phase 4) should use the *filtered* list of `ImageMetadata`.

**Step 5.5: Final QA (Visual Sweep)** (Plan 7.4)
    *   **Task**: Create a utility script that generates an HTML page or a PDF displaying all *kept* images side-by-side for a quick visual check.

---

### Phase 6: Leverage LaTeX Source Inputs (Corresponds to Plan Section 8)

**Goal**: If LaTeX sources are available, use them to get original, high-quality images and TikZ/PSTricks figures. This is a significant addition and might be treated as a separate feature branch or later enhancement if the project focuses first on pure PDF processing.

**Step 6.1: Scan for `\includegraphics`** (Plan 8.1)
    *   **Task**: If a `.tex` source directory is provided:
        *   Grep/regex search for `\includegraphics[*]{path/to/image}`.
        *   Copy these image files to `output/extracted_images/{pdf_name}/from_tex_source/`.
        *   Create `ImageMetadata` entries: `source_type: "raw_embedded"` (or "tex_includegraphics"), `page_number`: tricky, might need manual mapping or sophisticated analysis of `.aux` files. Bbox also tricky from source alone. This might be simpler if we just aim to collect all source images.

**Step 6.2: Extract Standalone TikZ/PSTricks** (Plan 8.2)
    *   **Task**:
        1.  Search for `\begin{tikzpicture}...\end{tikzpicture}` and `\begin{pspicture}...\end{pspicture}`.
        2.  For each, create a temporary minimal `.tex` file (e.g., using `standalone` class).
        3.  Batch compile these temp files to individual PDFs (then optionally to PNGs using `pdftoppm` or PyMuPDF).
        4.  Store these in `output/extracted_images/{pdf_name}/from_tex_source/tikz/`.
        5.  Create `ImageMetadata` entries. Page/bbox info is still the challenge.

**Step 6.3: Record Provenance and Deduplication** (Plan 8.3)
    *   **Task**:
        *   Add a `provenance: Literal["pdf_extracted", "tex_source"]` field to `ImageMetadata`.
        *   When merging images from PDF extraction and LaTeX source extraction:
            *   If an image from LaTeX source seems to be a duplicate of one extracted from the PDF (e.g., similar filename, or if page/bbox could be matched, or image content hash), prioritize the LaTeX source version.

---

### Phase 7: Documentation (Corresponds to Plan Section 10)

**Goal**: Update project documentation.

**Step 7.1: Update README.md** (Plan 10.1)
    *   **Task**:
        *   Describe the new multi-stage pipeline:
            1.  Image & Vector Extraction (from PDF and/or TeX).
            2.  Image Post-Processing/Filtering.
            3.  Text-Only PDF Generation.
            4.  OCR (on Text-Only PDF).
            5.  Markdown Assembly (merging OCR text with extracted images).
        *   Detail the new directory structures for extracted images, metadata, and text-only PDFs.
        *   Update setup instructions if new dependencies or configuration are needed.
        *   Explain how to use any new scripts or options.

---

**General Considerations**:
*   **Configuration**: Make thresholds (min image area, overlap IoU), paths, etc., configurable (e.g., via a central config file or command-line arguments).
*   **Logging**: Implement comprehensive logging throughout all stages. The current `mistral_ocr.py` has good logging setup that can be extended.
*   **Error Handling**: Robust error handling for file operations, API calls, and processing steps.
*   **Modularity**: Design each phase/step as a callable module/function to allow flexibility in orchestrating the pipeline.
*   **Refactoring Existing Code**: `pdf_ocr/mistral_ocr.py` will need significant refactoring. Identify parts that can be reused (like API client setup, argument parsing) and parts that need to be replaced or heavily modified (image handling, markdown generation).
