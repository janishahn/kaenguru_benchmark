# Känguru Benchmark

This project aims to build a Large Language Model (LLM) benchmark based on the Math Kangaroo competition, providing a challenging and diverse set of mathematical problems for evaluating LLM capabilities.

---

## Requirements

- Python 3.8+
- [mistralai](https://pypi.org/project/mistralai/) (for PDF OCR pipeline)
- [tqdm](https://pypi.org/project/tqdm/) (optional, for progress bars)
- [python-dotenv](https://pypi.org/project/python-dotenv/) (optional, for loading environment variables)
- [pandas](https://pypi.org/project/pandas/) (for Markdown to MMMU conversion)
- [pyarrow](https://pypi.org/project/pyarrow/) (optional, for Parquet index building)

To install all required packages:

```bash
pip install mistralai tqdm python-dotenv pandas pyarrow
```

---

## PDF OCR Pipeline

The project includes a PDF OCR pipeline for extracting text and images from Math Kangaroo competition PDFs using the Mistral AI OCR API. This pipeline automates the conversion of scanned or digital PDFs into structured Markdown, JSON, and image files for further processing and benchmarking.

### How It Works

1. **Input**: Accepts a single PDF file or a directory containing multiple PDF files.
2. **OCR Processing**: Each PDF is uploaded to the Mistral API, which performs OCR and returns structured results, including detected text and embedded images.
3. **Output Generation**:
    - **JSON**: The raw OCR response is saved as a JSON file.
    - **Images**: All images detected in the PDF are extracted and saved as separate files.
    - **Markdown**: The OCR text (with image references) is formatted into a Markdown file, with image links pointing to the extracted images.
4. **Batch Support**: The pipeline can process multiple PDFs in a directory sequentially.

### Output Structure

All outputs are saved in an `output/` directory at the project root, with the following subfolders:
- `ocr_json/`: Raw OCR JSON responses.
- `ocr_images/`: Extracted images from PDFs.
- `ocr_markdown/`: Markdown files with text and image references.
- `logs/`: Log files for each run.

### Setup

1. **Set your Mistral API key**:
    - Export it as an environment variable:
      ```bash
      export MISTRAL_API_KEY=your_api_key_here
      ```
    - Or create a `.env` file in the project root with:
      ```
      MISTRAL_API_KEY=your_api_key_here
      ```

### Usage

#### Process a Single PDF

```bash
python pdf_ocr/mistral_ocr.py path/to/your_file.pdf
```

#### Process All PDFs in a Directory

```bash
python pdf_ocr/mistral_ocr.py path/to/pdf_directory/
```

#### Enable Verbose Logging

```bash
python pdf_ocr/mistral_ocr.py path/to/your_file.pdf --verbose
```

#### Example Output

After running the script, you will find:
- `output/ocr_json/your_file_ocr_result.json`
- `output/ocr_markdown/your_file_ocr_result.md`
- Extracted images in `output/ocr_images/`
- Logs in `output/logs/`

---

## Markdown to MMMU Format Conversion

The project includes a script for converting OCR-generated Markdown files of Math Kangaroo exams and their solutions into the [MMMU](https://mmmu-benchmark.github.io/) (Massive Multi-discipline Multi-choice Understanding) JSON format, suitable for LLM benchmarking and dataset creation.

### How It Works

1. **Input**: The script expects a directory containing:
    - One or more exam Markdown files (named like `00_34_ocr_result.md`, `02_1113_ocr_result.md`, etc.)
    - A single solution Markdown file named `kaenguru_loesungen_alle_ocr_result.md` in the same directory
    - An `ocr_images/` directory (sibling to the Markdown files) containing all referenced images
2. **Parsing Solutions**: The script parses the solution Markdown, extracting answer keys for each year and grade span (e.g., `3-4`, `11-13`).
3. **Parsing Exams**: For each exam Markdown file:
    - Extracts year and grade range from the filename
    - Parses each question, its options, and any referenced images
    - Matches each question to its answer from the solution file
    - Calculates a difficulty score based on grade and point value
    - Replaces Markdown image tags with `<image N>` placeholders and collects image paths
4. **Output Generation**:
    - For each question, generates a single JSON file in `dataset/{year}/MathKangaroo_{year}_{grade_min}-{grade_max}_{question_code}.json`
    - Optionally, builds a Parquet index of all questions if `--build-index` is specified

### Input Format

- **Exam Markdown**: Each file represents one exam for a specific year and grade span. Questions are grouped by point value and formatted with options and images in Markdown.
- **Solution Markdown**: Contains tables of answers for each year and grade span, with task numbers and corresponding answer letters.
- **Images**: All images referenced in the Markdown must be present in the `ocr_images/` directory.

### Output Format

Each question is saved as a JSON file with the following structure:

```json
{
  "id": "MathKangaroo_2000_3-4_A1",
  "question_type": "multiple-choice",
  "question": "What is 2+2?",
  "options": ["(A) 3", "(B) 4", "(C) 5", "(D) 6", "(E) 7"],
  "answer": "B",
  "image_paths": ["/absolute/path/to/image1.png"],
  "grade_level_raw": "3-4",
  "grade_level_min": 3,
  "grade_level_max": 4,
  "year": 2000,
  "points": 3,
  "question_number_raw": "A1",
  "question_difficulty": 0.23
}
```

- Images in the question or options are replaced with `<image N>` placeholders, and their absolute paths are listed in `image_paths`.
- The `answer` field is filled from the solution file.
- The `question_difficulty` is a normalized score based on grade and points.

### Difficulty Score

Each question's `question_difficulty` (0–1) reflects its relative challenge, combining grade level (60%) and point value (40%), each normalized to a 0–1 scale: `difficulty = 0.6 * normalized_grade + 0.4 * normalized_points`. Higher grades and points yield higher difficulty. See `calculate_difficulty` in [`process_markdown/md_to_mmmu.py`](process_markdown/md_to_mmmu.py) for details.

### Setup

1. **Prepare your input directory**:
    - Place all exam Markdown files and the solution Markdown in the same folder
    - Ensure all referenced images are in an `ocr_images/` directory at the same level

### Usage

#### Convert Markdown to MMMU JSON

```bash
python process_markdown/md_to_mmmu.py path/to/your_markdown_dir/
```

#### Build a Parquet Index (optional)

```bash
python process_markdown/md_to_mmmu.py path/to/your_markdown_dir/ --build-index
```

#### Output

- Individual question JSON files in `dataset/{year}/`
- If `--build-index` is used: a single `math_kangaroo_dataset.parquet` file in `dataset/`
- Log file: `process_markdown/md_to_mmmu.log`

### Example Directory Structure

```
process_markdown/
  md_to_mmmu.py
  md_to_mmmu.log
ocr_output/
  ocr_images/
    00_34_page_1_img_1.jpg
    00_34_page_2_img_1.jpg
  ocr_markdown/
    00_34_ocr_result.md
    00_56_ocr_result.md
    kaenguru_loesungen_alle_ocr_result.md
```

### Notes
- The script is robust to minor formatting inconsistencies but expects the general structure described above.
- All logs are written to `process_markdown/md_to_mmmu.log` for debugging and traceability.
- The MMMU format is compatible with the [OpenCompass MMMU benchmark](https://github.com/MMMU-Benchmark/MMMU).
