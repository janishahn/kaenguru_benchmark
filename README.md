# Känguru Benchmark

This project aims to build a Large Language Model (LLM) benchmark based on the Math Kangaroo competition, providing a challenging and diverse set of mathematical problems for evaluating LLM capabilities.

---

## Requirements

- Python 3.8+
- [mistralai](https://pypi.org/project/mistralai/) (for PDF OCR pipeline)
- [tqdm](https://pypi.org/project/tqdm/) (optional, for progress bars)
- [python-dotenv](https://pypi.org/project/python-dotenv/) (for loading environment variables)
- [pandas](https://pypi.org/project/pandas/) (for data processing and analysis)
- [pyarrow](https://pypi.org/project/pyarrow/) (for Parquet file handling)
- [aiohttp](https://pypi.org/project/aiohttp/) (for asynchronous API requests)

To install all required packages:

```bash
pip install -r requirements.txt
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
- The MMMU format is compatible with the [MMMU benchmark](https://github.com/MMMU-Benchmark/MMMU).

---

## LLM Inference

The project includes a comprehensive LLM inference script for evaluating language models against the Math Kangaroo benchmark. The script handles prompt construction, efficient batched API requests, answer extraction, and detailed performance analysis.

### How It Works

1. **Input**: Takes a path to a Kangaroo dataset in Parquet format.
2. **Preprocessing**:
    - Loads question data from the specified Parquet file.
    - Filters out multimodal items that contain images (optional).
    - Validates required columns in the input data.
3. **Prompt Construction**:
    - Constructs prompts for each question, including question text, options, and a prompt template.
    - Supports two prompt types: reasoning (for step-by-step problem solving) and no-reasoning (for direct answers).
4. **API Requests**:
    - Makes asynchronous requests to the OpenRouter API with configurable concurrency and batch size.
    - Implements exponential backoff retry logic for handling API errors and rate limits.
    - Records detailed information about each request, including latency and response.
5. **Answer Extraction**:
    - Extracts answers from LLM responses using robust regex patterns.
    - Handles various answer formats (e.g., "A", "(A)", "The answer is A").
    - Detects and handles truncated responses.
    - Compares extracted answers with ground truth.
6. **Output Generation**:
    - Generates comprehensive metrics including overall and per-year accuracy.
    - Saves results in multiple formats for analysis and inspection.
    - Provides detailed console output with progress updates.

### Setup

1. **Set your OpenRouter API key**:
    - Create a `.env` file in the project root with:
      ```
      OPENROUTER_API_KEY=your_api_key_here
      ```

2. **Create prompt templates**:
    - Create two prompt template files in `llm_inference/prompts/`:
      - `reasoning_prompt.md`: Template for step-by-step reasoning.
      - `no_reasoning_prompt.md`: Template for direct answers.

### Usage

#### Basic Usage

```bash
python llm_inference/llm_inference.py path/to/dataset.parquet
```

#### Specify Model and Output Directory

```bash
python llm_inference/llm_inference.py path/to/dataset.parquet --model anthropic/claude-3-opus-20240229 --output_dir outputs/claude3
```

#### Configure Batch Processing

```bash
python llm_inference/llm_inference.py path/to/dataset.parquet --batch_size 5 --concurrency 3
```

#### Enable Reasoning Mode

```bash
python llm_inference/llm_inference.py path/to/dataset.parquet --reasoning
```

#### Configure Token Limits

```bash
python llm_inference/llm_inference.py path/to/dataset.parquet --max_tokens 8000
```

#### Enable Debug Logging (Log Files Only)

```bash
python llm_inference/llm_inference.py path/to/dataset.parquet --debug
```

This will write detailed debug-level logs to the log files (`llm_inference.log` and `preprocessing.log`), but the console will still only show warnings and above.

### Command Line Arguments

- `input_file` (required): Path to Kangaroo dataset .parquet file
- `--model`: OpenRouter model slug (default: "google/gemini-2.5-flash-preview")
- `--output-dir`: Directory to save output files (default: "llm_outputs")
- `--batch-size`: Number of samples to process in each batch (default: 1)
- `--max-retries`: Maximum number of retries for failed API calls (default: 3)
- `--concurrency`: Number of concurrent API calls (default: 1)
- `--reasoning`: Enable reasoning mode for step-by-step problem solving
- `--max-tokens`: Maximum number of tokens to generate (default: 4000, increased for reasoning)
- `--debug`: Enable debug-level logging to log files (not console)

### Output Structure

All outputs are saved in timestamped directories in the specified output directory:
- `results.parquet`: Processed results with model answers and correctness flags
- `metrics.json`: Comprehensive metrics including overall and per-year accuracy
- `raw_responses.jsonl`: Full API request and response data for analysis
- `batch_xxx_results.json`: Individual batch results
- `all_results.json`: All API responses combined
- `prompts.txt`: Generated prompts for inspection
- `config.txt`: Run configuration details
- `preprocessing.log`: Detailed logs
- `filtered_dataset.parquet`: Dataset after filtering out multimodal items

### Metrics and Evaluation

The script evaluates model performance with several metrics:
- Overall accuracy: Percentage of correctly answered questions
- Per-year accuracy: Breakdown of accuracy by competition year
- Latency statistics: Average, P95, and P99 response times
- Success rate: Percentage of successful API calls
- Error rate: Percentage of questions that resulted in errors
- Truncation rate: Percentage of responses that were truncated
- A detailed console summary is printed at the end of each run

### Example Output

The `metrics.json` file contains comprehensive statistics:

```json
{
  "overall": {
    "total_questions": 100,
    "answered_questions": 95,
    "correct_answers": 80,
    "error_answers": 5,
    "truncated_responses": 2,
    "accuracy": 0.842,
    "average_latency_ms": 1234.5,
    "p95_latency_ms": 2000.0,
    "p99_latency_ms": 3000.0,
    "success_rate": 0.95
  },
  "per_year": {
    "2020": {
      "total_questions": 30,
      "answered_questions": 28,
      "correct_answers": 24,
      "truncated_responses": 1,
      "accuracy": 0.857,
      "average_latency_ms": 1200.0,
      "success_rate": 0.93
    }
    // ... other years
  }
}
```
