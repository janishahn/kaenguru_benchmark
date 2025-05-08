#!/usr/bin/env python3

import argparse
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple
import time
import json
import asyncio
import aiohttp
import re
from collections import Counter
from dotenv import load_dotenv
from tqdm import tqdm
import traceback

# Required columns in the input parquet file
REQUIRED_COLUMNS = {
    'id': 'Unique identifier for each question',
    'image_paths': 'List of image paths (empty for text-only questions)',
    'question': 'The question text',
    'options': 'List of answer options (A-E)',
    'answer': 'The correct answer (A-E)',
    'year': 'Year of the question (for per-year metrics)'
}


def setup_logging(output_dir: Path, debug: bool = False) -> None:
    """Set up logging configuration."""
    # Create log file in the script's directory
    script_dir = Path(__file__).parent
    script_log_file = script_dir / "llm_inference.log"
    
    # Create log file in the output directory for this run
    run_log_file = output_dir / "preprocessing.log"
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create and configure file handlers
    script_handler = logging.FileHandler(script_log_file, mode='a')  # Append mode
    script_handler.setFormatter(formatter)
    script_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    
    run_handler = logging.FileHandler(run_log_file, mode='w')  # Write mode (overwrite)
    run_handler.setFormatter(formatter)
    run_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Create and configure console handler - only show WARNING and above
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.addHandler(script_handler)
    root_logger.addHandler(run_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial messages to files only
    logging.info(f"Logging to script log file: {script_log_file}")
    logging.info(f"Logging to run log file: {run_log_file}")


def load_prompt(prompt_type: str) -> str:
    """Load prompt from file."""
    prompt_path = Path(__file__).parent / "prompts" / f"{prompt_type}_prompt.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, "r") as f:
        return f.read().strip()


def load_question_data(input_path: Path) -> pd.DataFrame:
    """Load question data from parquet file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.suffix == '.parquet':
        raise ValueError(f"Input file must be a parquet file: {input_path}")
    
    logging.info(f"Loading question data from: {input_path}")
    
    # Load the data
    df = pd.read_parquet(input_path)
    
    # Validate required columns
    missing_columns = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing_columns:
        error_msg = "Missing required columns in input file:\n"
        for col in missing_columns:
            error_msg += f"- {col}: {REQUIRED_COLUMNS[col]}\n"
        raise ValueError(error_msg)
    
    # Log column information
    logging.info("Input file columns:")
    for col in df.columns:
        if col in REQUIRED_COLUMNS:
            logging.info(f"- {col}: {REQUIRED_COLUMNS[col]}")
        else:
            logging.info(f"- {col}: Additional column")
    
    return df


def filter_multimodal_items(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out items that contain images."""
    initial_count = len(df)
    logging.info(f"Initial dataset size: {initial_count} items")
    
    def has_images(x):
        # Handle None case
        if x is None:
            return False
            
        # Handle string representation of list
        if isinstance(x, str):
            try:
                # Try to evaluate as a list if it looks like a string representation
                if x.startswith('[') and x.endswith(']'):
                    x = eval(x)
                else:
                    return False
            except:
                return False
                
        # Handle actual list
        if isinstance(x, list):
            return len(x) > 0
            
        # Handle other sequence types
        if hasattr(x, '__len__'):
            return len(x) > 0
            
        # Log unexpected type
        logging.warning(f"Unexpected type for image_paths: {type(x)}")
        return False
    
    # Filter rows where image_paths is empty
    filtered_df = df[~df['image_paths'].apply(has_images)].copy()
    
    final_count = len(filtered_df)
    logging.info(f"Filtered dataset size: {final_count} items")
    logging.info(f"Removed {initial_count - final_count} multimodal items")
    
    # Log some examples of filtered items for debugging
    if initial_count - final_count > 0:
        sample_filtered = df[df['image_paths'].apply(has_images)].head(3)
        for _, row in sample_filtered.iterrows():
            logging.info(f"Filtered item example - ID: {row['id']}, image_paths type: {type(row['image_paths'])}, value: {row['image_paths']}")
    
    return filtered_df


def construct_prompts(df: pd.DataFrame, prompt_type: str) -> List[str]:
    """Construct prompts for each question."""
    # Load the appropriate prompt
    prompt_template = load_prompt(prompt_type)
    logging.info(f"Loaded {prompt_type} prompt template")
    
    messages = []
    for _, row in df.iterrows():
        # Construct prompt following MMMU-style format
        prompt = (
            f"[ID: {row['id']}]\n"
            # TODO: decide if we want to include the grade level and year in the prompt
            # f"Grade level {row['grade_level_raw']}  Year {row['year']}  Points {row['points']}\n\n"
            f"Question:\n{row['question']}\n\n"
            f"Options:\n"
            f"{'  '.join(row['options'])}\n\n"
            f"{prompt_template}"
        )
        
        messages.append(prompt)
    
    return messages


def create_output_dir(base_output_dir: Path) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / f"run-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    return output_dir


def get_openrouter_headers() -> Dict[str, str]:
    """Get headers for OpenRouter API requests."""
    # Load environment variables from .env file
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    return headers


def prepare_request_body(model: str, prompt: str, max_tokens: int, is_reasoning: bool) -> Dict[str, Any]:
    """Prepare request body for OpenRouter API."""
    # Set minimum token limits based on reasoning mode
    min_tokens = 15000 if is_reasoning else 10000
    # Use the larger of user-specified max_tokens or minimum required tokens
    effective_max_tokens = max(max_tokens, min_tokens)
    
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": effective_max_tokens,
        "usage": {"include": True},  # Include usage information in response
        # "seed": 42,  # Random seed for reproducibility
        # "response_format": {"type": "text"},  # Force text response format
    }


async def make_openrouter_request_async(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    max_retries: int,
    max_tokens: int,
    is_reasoning: bool,
    semaphore: asyncio.Semaphore,
    attempt: int = 1
) -> Tuple[Dict[str, Any], float, int]:
    """Make asynchronous request to OpenRouter API with retry logic."""
    async with semaphore:  # Acquire semaphore before making request
        headers = get_openrouter_headers()
        body = prepare_request_body(model, prompt, max_tokens, is_reasoning)
        
        start_time = time.time()
        
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=body
            ) as response:
                response_json = await response.json()
                latency = time.time() - start_time
                
                if response.status >= 500 or response.status == 429:  # Rate limit
                    if attempt < max_retries:
                        wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60 seconds
                        logging.debug(f"API request error (attempt {attempt}/{max_retries}): Status {response.status}, Prompt ID: {prompt[:100]}...")
                        logging.debug(f"Response body: {response_json}")
                        logging.debug(f"Retrying in {wait_time} seconds...")
                        logging.debug(f"API request failed (attempt {attempt}/{max_retries}). Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        return await make_openrouter_request_async(
                            session, prompt, model, max_retries, max_tokens, is_reasoning, semaphore, attempt + 1
                        )
                elif response.status != 200:
                    logging.debug(f"API request non-200 status: {response.status}, Prompt ID: {prompt[:100]}...")
                    logging.debug(f"Response body: {response_json}")
                
                return {
                    "request": body,
                    "response": response_json,
                    "latency": latency,
                    "status_code": response.status
                }
                
        except Exception as e:
            logging.debug(f"Exception during API request (attempt {attempt}/{max_retries}), Prompt ID: {prompt[:100]}...")
            logging.debug(f"Exception type: {type(e).__name__}, Message: {e}")
            logging.debug(traceback.format_exc())
            if attempt < max_retries:
                wait_time = min(2 ** attempt, 60)
                logging.debug(f"API request failed due to exception (attempt {attempt}/{max_retries}). Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                return await make_openrouter_request_async(
                    session, prompt, model, max_retries, max_tokens, is_reasoning, semaphore, attempt + 1
                )
            raise e


async def process_all_prompts(
    prompts: List[str],
    model: str,
    max_retries: int,
    output_dir: Path,
    max_tokens: int,
    is_reasoning: bool,
    concurrency: int
) -> List[Dict[str, Any]]:
    """Process all prompts asynchronously with controlled concurrency (no batching)."""
    all_results = []
    pbar = tqdm(total=len(prompts), desc="Processing questions", unit="q")
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            make_openrouter_request_async(
                session, prompt, model, max_retries, max_tokens, is_reasoning, semaphore
            )
            for prompt in prompts
        ]
        for i, future in enumerate(asyncio.as_completed(tasks), 1):
            result = await future
            all_results.append(result)
            pbar.update(1)
            # Optionally save progress every 50 results
            if i % 50 == 0 or i == len(prompts):
                progress_file = output_dir / "all_results.json"
                with open(progress_file, "w") as f:
                    json.dump(all_results, f, indent=2)
    pbar.close()
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM inference script for Kangaroo benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required positional argument
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to Kangaroo .parquet file (relative to working directory)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.5-flash-preview",  # Default to Gemini 2.5 Flash
        help="OpenRouter model slug to use"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llm_outputs",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples to process in each batch"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed API calls"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent API calls"
    )

    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Use reasoning prompt for step-by-step problem solving"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum number of tokens to generate in response (will be increased for reasoning)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to log files (not console)"
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    working_dir = Path.cwd()
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_path = working_dir / args.input_file
    output_path = project_root / args.output_dir
    
    # Validate input file
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")
    
    if not input_path.suffix == '.parquet':
        raise ValueError(f"Input file must be a parquet file: {input_path}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Update paths in args to be absolute
    args.input_file = str(input_path)
    args.output_dir = str(output_path)
    
    return args


def extract_answer_from_response(response: Dict[str, Any]) -> str:
    """Extract answer from model response using MMMU strategy with enhanced robustness."""
    try:
        # Check if response was truncated due to token limit
        finish_reason = response["choices"][0].get("finish_reason")
        if finish_reason == "length":
            logging.warning("Response was truncated due to token limit")
            return "ERROR"
            
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return "ERROR"
    
    # Try to find the answer at the end of the text first (priority for reasoning outputs)
    # Look for patterns like "The answer is X" or "Therefore, X" near the end
    final_answer_patterns = [
        r'(?:answer\s+is|therefore|thus|so)\s+(?:option\s+)?[\"\'(]?([A-E])[\"\')]?(?:\.|$)',
        r'(?:I\s+(?:will\s+)?(?:choose|select|pick|go\s+with))\s+(?:option\s+)?[\"\'(]?([A-E])[\"\')]?(?:\.|$)',
        r'(?:final\s+answer|conclusion)(?:\s+is)?\s*(?::|->)?\s*[\"\'(]?([A-E])[\"\')]?(?:\.|$)'
    ]
    
    # Check the last 20% of the text for conclusion patterns
    last_part = content[int(len(content) * 0.8):]
    for pattern in final_answer_patterns:
        match = re.search(pattern, last_part, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Look for a standalone answer format like "Answer: X" anywhere
    answer_format = re.search(r'(?:^|[\n\r])(?:answer|option)(?:\s+is)?[\s:]+[\"\'(]?([A-E])[\"\')]?(?:\.|$)', content, re.IGNORECASE | re.MULTILINE)
    if answer_format:
        return answer_format.group(1)
    
    # Find all options-like patterns and take the last one (reasoning usually ends with the answer)
    option_patterns = re.findall(r'(?:^|[\s,.])(?:option\s+)?[\"\'(]?([A-E])[\"\')]?(?:[\s,.]|$)', content)
    if option_patterns:
        return option_patterns[-1]
    
    # Original MMMU approach as fallback
    # First try: Look for a single capital letter A-E
    letter_match = re.search(r'\b([A-E])\b', content)
    if letter_match:
        return letter_match.group(1)
    
    # Second try: Look for any capital letter A-E
    letter_match = re.search(r'[A-E]', content)
    if letter_match:
        return letter_match.group(0)
    
    # Fallback: Count all A-E letters and take most frequent
    letters = re.findall(r'[A-E]', content)
    if letters:
        return Counter(letters).most_common(1)[0][0]
    
    return "ERROR"


def process_results_with_progress(
    results: List[Dict[str, Any]],
    df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Process results and add model answers to dataframe with progress bar."""
    # Create progress bar for processing results
    pbar = tqdm(total=len(results), desc="Processing results", unit="result")
    
    # Create a mapping of question IDs to their indices
    id_to_idx = {row['id']: idx for idx, row in df.iterrows()}
    
    # Initialize model_answer column
    df['model_answer'] = None
    df['is_correct'] = None
    df['response_latency'] = 0.0  # Initialize with 0.0 instead of None
    df['response_status'] = None
    df['was_truncated'] = None
    df['generation_cost'] = 0.0  # Initialize cost tracking
    df['input_tokens'] = 0  # Track prompt tokens
    df['output_tokens'] = 0  # Track completion tokens
    
    # Process each result
    for result in results:
        try:
            # Extract question ID from the prompt
            prompt = result['request']['messages'][0]['content']
            id_match = re.search(r'\[ID: ([^\]]+)\]', prompt)
            if not id_match:
                pbar.update(1)
                continue
            
            question_id = id_match.group(1)
            if question_id not in id_to_idx:
                pbar.update(1)
                continue
            
            idx = id_to_idx[question_id]
            
            # Check if response was truncated
            was_truncated = result['response']['choices'][0].get('finish_reason') == 'length'
            df.at[idx, 'was_truncated'] = was_truncated
            
            # Extract answer and update dataframe
            model_answer = extract_answer_from_response(result['response'])
            df.at[idx, 'model_answer'] = model_answer
            df.at[idx, 'is_correct'] = model_answer == df.at[idx, 'answer']
            
            # Ensure latency is properly set
            if 'latency' in result and isinstance(result['latency'], (int, float)):
                df.at[idx, 'response_latency'] = float(result['latency'])
            else:
                logging.warning(f"Invalid or missing latency for question {question_id}")
                df.at[idx, 'response_latency'] = 0.0
            
            df.at[idx, 'response_status'] = result['status_code']
            
            # Extract generation cost if available
            if 'usage' in result['response'] and 'cost' in result['response']['usage']:
                df.at[idx, 'generation_cost'] = float(result['response']['usage']['cost'])
            else:
                logging.warning(f"Missing cost information for question {question_id}")
                df.at[idx, 'generation_cost'] = 0.0
            
            # Extract token usage if available
            if 'usage' in result['response']:
                usage = result['response']['usage']
                df.at[idx, 'input_tokens'] = int(usage.get('prompt_tokens', 0))
                df.at[idx, 'output_tokens'] = int(usage.get('completion_tokens', 0))
            else:
                df.at[idx, 'input_tokens'] = 0
                df.at[idx, 'output_tokens'] = 0
            
        except Exception as e:
            logging.error(f"Error processing result: {e}")
        finally:
            pbar.update(1)
    
    pbar.close()
    
    # Calculate statistics
    total_questions = len(df)
    answered_questions = df['model_answer'].notna().sum()
    correct_answers = df['is_correct'].sum()
    error_answers = (df['model_answer'] == 'ERROR').sum()
    truncated_responses = df['was_truncated'].sum()
    total_cost = df['generation_cost'].sum()
    
    # Log statistics
    logging.info(f"Total questions: {total_questions}")
    logging.info(f"Answered questions: {answered_questions}")
    logging.info(f"Correct answers: {correct_answers}")
    logging.info(f"Error answers: {error_answers}")
    logging.info(f"Truncated responses: {truncated_responses}")
    logging.info(f"Total generation cost: {total_cost:.2f} dollars")
    logging.info(f"Accuracy: {correct_answers/answered_questions:.2%}" if answered_questions > 0 else "Accuracy: N/A")
    
    # Save processed results
    results_path = output_dir / "processed_results.parquet"
    df.to_parquet(results_path)
    logging.info(f"Saved processed results to: {results_path}")
    
    return df


def generate_human_readable_report(
    df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate a human-readable report of questions, LLM answers, and correct answers."""
    # Create a CSV file for easy viewing in spreadsheet software
    csv_path = output_dir / "question_answers_overview.csv"
    
    # Select and rename columns for the report
    report_df = df[['id', 'question', 'options', 'model_answer', 'answer', 'is_correct', 'year', 'was_truncated', 'response_latency']].copy()
    report_df = report_df.rename(columns={
        'id': 'Question ID',
        'question': 'Question',
        'options': 'Options',
        'model_answer': 'LLM Answer',
        'answer': 'Correct Answer',
        'is_correct': 'Is Correct',
        'year': 'Year',
        'was_truncated': 'Was Truncated',
        'response_latency': 'Response Time (s)'
    })
    
    # Convert options list to a formatted string for readability
    report_df['Options'] = report_df['Options'].apply(lambda opts: '\n'.join(opts))
    
    # Save to CSV
    report_df.to_csv(csv_path, index=False)
    logging.info(f"Saved human-readable overview to: {csv_path}")
    
    # Also create a more readable text file version
    txt_path = output_dir / "question_answers_overview.txt"
    with open(txt_path, "w") as f:
        f.write("=== QUESTIONS AND ANSWERS OVERVIEW ===\n\n")
        
        # Group by year for better organization
        for year, year_group in df.groupby('year'):
            f.write(f"== YEAR {int(year)} ==\n")
            f.write(f"Total questions: {len(year_group)}\n")
            f.write(f"Correct answers: {year_group['is_correct'].sum()} ({year_group['is_correct'].mean():.2%})\n\n")
            
            # Write each question and answer
            for _, row in year_group.iterrows():
                f.write(f"ID: {row['id']}\n")
                f.write(f"Question: {row['question']}\n")
                f.write("Options:\n")
                for opt in row['options']:
                    f.write(f"  {opt}\n")
                f.write(f"LLM Answer: {row['model_answer']}\n")
                f.write(f"Correct Answer: {row['answer']}\n")
                f.write(f"Correct: {'✓' if row['is_correct'] else '✗'}\n")
                latency = row['response_latency']
                if pd.isna(latency) or not isinstance(latency, (int, float)):
                    f.write("Response time: N/A\n")
                else:
                    f.write(f"Response time: {float(latency):.2f}s\n")
                f.write(f"Truncated: {'Yes' if row['was_truncated'] else 'No'}\n")
                f.write("\n" + "-"*50 + "\n\n")
        
        # Add overall statistics at the end
        f.write("=== OVERALL STATISTICS ===\n")
        f.write(f"Total questions: {len(df)}\n")
        f.write(f"Correct answers: {df['is_correct'].sum()} ({df['is_correct'].mean():.2%})\n")
        # Calculate average response time only from valid values
        valid_latencies = df['response_latency'].dropna()
        if len(valid_latencies) > 0:
            f.write(f"Average response time: {valid_latencies.mean():.2f}s\n")
        else:
            f.write("Average response time: N/A\n")
        f.write(f"Truncated responses: {df['was_truncated'].sum()} ({df['was_truncated'].mean():.2%})\n")
    
    logging.info(f"Saved detailed text overview to: {txt_path}")


def generate_final_outputs_with_progress(
    df: pd.DataFrame,
    results: List[Dict[str, Any]],
    output_dir: Path
) -> Tuple[Dict[str, Any], List[str]]:
    """Generate final output files and metrics with progress bar."""
    # Create progress bar for generating outputs
    pbar = tqdm(total=5, desc="Generating final outputs", unit="file")
    
    # 1. Generate results.parquet
    results_df = df.copy()
    results_df['latency_ms'] = results_df['response_latency'] * 1000  # Convert to milliseconds
    results_df = results_df.drop(columns=['response_latency', 'response_status'])
    results_path = output_dir / "results.parquet"
    results_df.to_parquet(results_path)
    logging.info(f"Saved results to: {results_path}")
    pbar.update(1)
    
    # 2. Generate metrics.json
    total_cost = float(df['generation_cost'].sum())
    # Robust success rate calculation: only consider rows with non-null response_status
    valid_status = df['response_status'].notna()
    if valid_status.any():
        success_rate = float((df.loc[valid_status, 'response_status'] == 200).mean())
    else:
        success_rate = None
    metrics = {
        "overall": {
            "total_questions": int(len(df)),
            "answered_questions": int(df['model_answer'].notna().sum()),
            "correct_answers": int(df['is_correct'].sum()),
            "error_answers": int((df['model_answer'] == 'ERROR').sum()),
            "truncated_responses": int(df['was_truncated'].sum()),
            "accuracy": float(df['is_correct'].mean()) if df['is_correct'].notna().any() else None,
            "average_latency_ms": float(df['response_latency'].mean() * 1000),
            "p95_latency_ms": float(df['response_latency'].quantile(0.95) * 1000),
            "p99_latency_ms": float(df['response_latency'].quantile(0.99) * 1000),
            "success_rate": success_rate,
            "total_cost_dollars": total_cost,
            "average_cost_per_question": float(df['generation_cost'].mean()),
            "total_input_tokens": int(df['input_tokens'].sum()),
            "total_output_tokens": int(df['output_tokens'].sum()),
            "total_tokens": int(df['input_tokens'].sum() + df['output_tokens'].sum()),
            "average_input_tokens": float(df['input_tokens'].mean()),
            "average_output_tokens": float(df['output_tokens'].mean())
        },
        "per_year": {}
    }
    
    # Calculate per-year metrics
    for year in df['year'].unique():
        year_df = df[df['year'] == year]
        year_metrics = {
            "total_questions": int(len(year_df)),
            "answered_questions": int(year_df['model_answer'].notna().sum()),
            "correct_answers": int(year_df['is_correct'].sum()),
            "truncated_responses": int(year_df['was_truncated'].sum()),
            "accuracy": float(year_df['is_correct'].mean()) if year_df['is_correct'].notna().any() else None,
            "average_latency_ms": float(year_df['response_latency'].mean() * 1000),
            "success_rate": float((year_df['response_status'] == 200).mean()),
            "total_cost_dollars": float(year_df['generation_cost'].sum()),
            "average_cost_per_question": float(year_df['generation_cost'].mean()),
            "total_input_tokens": int(year_df['input_tokens'].sum()),
            "total_output_tokens": int(year_df['output_tokens'].sum()),
            "total_tokens": int(year_df['input_tokens'].sum() + year_df['output_tokens'].sum()),
            "average_input_tokens": float(year_df['input_tokens'].mean()),
            "average_output_tokens": float(year_df['output_tokens'].mean())
        }
        metrics["per_year"][str(int(year))] = year_metrics
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to: {metrics_path}")
    pbar.update(1)
    
    # 3. Generate raw_responses.jsonl
    responses_path = output_dir / "raw_responses.jsonl"
    with open(responses_path, "w") as f:
        for result in results:
            try:
                # Extract question ID from the prompt
                prompt = result['request']['messages'][0]['content']
                id_match = re.search(r'\[ID: ([^\]]+)\]', prompt)
                if not id_match:
                    continue
                
                question_id = id_match.group(1)
                
                # Safely check for truncation
                was_truncated = False
                try:
                    if 'response' in result and 'choices' in result['response'] and len(result['response']['choices']) > 0:
                        was_truncated = result['response']['choices'][0].get('finish_reason') == 'length'
                except (KeyError, IndexError, TypeError):
                    logging.warning(f"Could not determine truncation status for question {question_id}")
                
                # Create response record
                response_record = {
                    "question_id": question_id,
                    "request": result['request'],
                    "response": result['response'],
                    "latency_ms": float(result['latency'] * 1000),
                    "status_code": int(result['status_code']),
                    "was_truncated": was_truncated,
                    "generation_cost": float(result['response']['usage']['cost']) if 'usage' in result['response'] and 'cost' in result['response']['usage'] else 0.0
                }
                
                f.write(json.dumps(response_record) + "\n")
            except Exception as e:
                logging.error(f"Error processing result for raw_responses.jsonl: {e}")
                continue
    
    logging.info(f"Saved raw responses to: {responses_path}")
    pbar.update(1)
    
    # 4. Generate human-readable report of questions and answers
    generate_human_readable_report(df, output_dir)
    pbar.update(1)
    
    # 5. (Summary printout removed; will be printed at the end of main_async)
    pbar.update(1)
    
    pbar.close()
    
    # Return metrics and output file paths for final summary
    return metrics, [
        output_dir / 'question_answers_overview.csv',
        output_dir / 'question_answers_overview.txt'
    ]


async def main_async():
    args = parse_args()
    
    # Calculate effective max tokens
    min_tokens = 15000 if args.reasoning else 10000
    effective_max_tokens = max(args.max_tokens, min_tokens)
    
    print("\n=== Starting LLM Inference ===")
    print(f"Model: {args.model}")
    print(f"Reasoning mode: {'Enabled' if args.reasoning else 'Disabled'}")
    print(f"Requested max tokens: {args.max_tokens}")
    print(f"Effective max tokens: {effective_max_tokens} (minimum: {min_tokens})")
    print(f"Batch size: {args.batch_size}")
    print(f"Concurrency: {args.concurrency}")
    print("=" * 30 + "\n")
    
    # Start timing the entire process
    start_time = time.time()
    
    # Create timestamped output directory
    output_dir = create_output_dir(Path(args.output_dir))
    
    # Set up logging
    setup_logging(output_dir, debug=args.debug)
    
    # Load question data
    try:
        df = load_question_data(Path(args.input_file))
        print(f"Successfully loaded {len(df)} questions from input file")
    except Exception as e:
        logging.error(f"Error loading question data: {e}")
        print(f"\nError: Failed to load question data: {e}")
        return
    
    # Filter out multimodal items
    filtered_df = filter_multimodal_items(df)
    print(f"Filtered to {len(filtered_df)} text-only questions")
    
    # Save filtered dataset
    filtered_path = output_dir / "filtered_dataset.parquet"
    filtered_df.to_parquet(filtered_path)
    logging.info(f"Saved filtered dataset to: {filtered_path}")
    
    # Construct prompts
    prompt_type = "reasoning" if args.reasoning else "no_reasoning"
    messages = construct_prompts(filtered_df, prompt_type)
    logging.info(f"Constructed {len(messages)} prompts")
    
    # Save prompts for inspection
    prompts_path = output_dir / "prompts.txt"
    with open(prompts_path, "w") as f:
        f.write("=== Prompts ===\n")
        for i, prompt in enumerate(messages):
            f.write(f"\n--- Prompt {i+1} ---\n")
            f.write(prompt)
            f.write("\n")
    logging.info(f"Saved prompts to: {prompts_path}")
    
    print("\nStarting inference...")
    # Process prompts in batches
    results = await process_all_prompts(
        messages,
        args.model,
        args.max_retries,
        output_dir,
        args.max_tokens,
        args.reasoning,
        args.concurrency
    )
    
    print("\nProcessing results...")
    # Process results and evaluate answers with progress bar
    processed_df = process_results_with_progress(results, filtered_df, output_dir)
    
    # Generate final outputs with progress bar
    metrics, overview_files = generate_final_outputs_with_progress(processed_df, results, output_dir)
    
    # Calculate total time taken
    total_time = time.time() - start_time
    avg_time_per_query = total_time / len(filtered_df) if not filtered_df.empty else 0
    
    # Save final configuration
    config = {
        "model": args.model,
        "batch_size": args.batch_size,
        "max_retries": args.max_retries,
        "concurrency": args.concurrency,
        "reasoning_enabled": args.reasoning,
        "max_tokens": args.max_tokens,
        "effective_max_tokens": effective_max_tokens,
        "timestamp": datetime.now().isoformat(),
        "initial_questions": len(df),
        "filtered_questions": len(filtered_df),
        "total_requests": len(results),
        "successful_requests": sum(1 for r in results if r["status_code"] == 200),
        "average_latency": sum(r["latency"] for r in results) / len(results) if results else 0,
        "accuracy": float(processed_df['is_correct'].mean()) if processed_df['is_correct'].notna().any() else None,
        "total_time_seconds": total_time,
        "average_time_per_query_seconds": avg_time_per_query,
        "total_cost_dollars": float(processed_df['generation_cost'].sum()),
        "average_cost_per_question": float(processed_df['generation_cost'].mean())
    }
    
    # Save configuration
    with open(output_dir / "config.txt", "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Max retries: {args.max_retries}")
    logging.info(f"Concurrency: {args.concurrency}")
    logging.info(f"Reasoning enabled: {args.reasoning}")
    logging.info(f"Max tokens: {args.max_tokens}")
    logging.info(f"Effective max tokens: {effective_max_tokens}")
    logging.info(f"Total requests: {config['total_requests']}")
    logging.info(f"Successful requests: {config['successful_requests']}")
    logging.info(f"Average latency: {config['average_latency']:.2f}s")
    logging.info(f"Total time: {str(timedelta(seconds=int(total_time)))}")
    logging.info(f"Average time per query: {avg_time_per_query:.2f}s")
    logging.info(f"Total cost: ${config['total_cost_dollars']:.3f}")
    logging.info(f"Average cost per question: ${config['average_cost_per_question']:.5f}")
    logging.info(f"Accuracy: {config['accuracy']:.2%}" if config['accuracy'] is not None else "Accuracy: N/A")
    
    # Print consolidated summary combining all statistics
    print("\n=== Inference Complete ===")
    print(f"Results saved to: {output_dir}")
    print("\n=== Summary ===")
    print(f"Total questions: {metrics['overall']['total_questions']}")
    print(f"Answered questions: {metrics['overall']['answered_questions']}")
    print(f"Correct answers: {metrics['overall']['correct_answers']}")
    print(f"Truncated responses: {metrics['overall']['truncated_responses']}")
    print(f"Overall accuracy: {metrics['overall']['accuracy']:.2%}")
    print(f"Total requests: {config['total_requests']}")
    print(f"Successful requests: {config['successful_requests']}/{config['total_requests']} ({metrics['overall']['success_rate']:.2%})")
    print(f"Average latency: {metrics['overall']['average_latency_ms']:.1f}ms")
    print(f"Total time taken: {str(timedelta(seconds=int(total_time)))}")
    print(f"Average time per query: {avg_time_per_query:.2f}s")
    print(f"Total cost: ${metrics['overall']['total_cost_dollars']:.3f}")
    print(f"Average cost per question: ${metrics['overall']['average_cost_per_question']:.5f}")
    # Print token stats in one line
    print(f"Token usage: avg input {metrics['overall']['average_input_tokens']:.1f}, avg output {metrics['overall']['average_output_tokens']:.1f}, total processed {metrics['overall']['total_tokens']}")
    print("=" * 30 + "\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
