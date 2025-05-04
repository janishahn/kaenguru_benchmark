# Kaenguru Benchmark

This project aims to build a Large Language Model (LLM) benchmark based on the Math Kangaroo competition, providing a challenging and diverse set of mathematical problems for evaluating LLM capabilities.

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

### Requirements

- Python 3.8+
- [mistralai](https://pypi.org/project/mistralai/)
- (Optional) [tqdm](https://pypi.org/project/tqdm/) for progress bars
- (Optional) [python-dotenv](https://pypi.org/project/python-dotenv/) for loading environment variables

### Setup

1. **Install dependencies**:
    ```bash
    pip install mistralai tqdm python-dotenv
    ```
2. **Set your Mistral API key**:
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
