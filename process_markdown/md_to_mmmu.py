#!/usr/bin/env python3

import argparse
import json
import logging
import re
from pathlib import Path
import pandas as pd
import math
from tqdm import tqdm # Import tqdm

# Constants
INDEX_FILENAME = "kangaroo_index.parquet"

# Setup logging
log_file_path = Path(__file__).parent / "md_to_mmmu.log"

# Configure file handler first
file_handler = logging.FileHandler(log_file_path, mode='w') # Added mode='w' to overwrite log on each run
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Configure console handler for errors only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # Only show errors on console
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

# Setup root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Capture all messages
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

OUTPUT_DIR = (Path(__file__).parent.parent / "dataset").resolve()
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# New function to parse filename
def parse_filename_info(filename: str) -> tuple[int | None, int | None, int | None]:
    """
    Parses year, grade_min, and grade_max from the exam filename.
    Filename structure: {year-abbreviation}_{grade-from}{grade-to}_ocr_result.md
    Examples:
    - 22_1113_ocr_result.md -> year=2022, grade_from=11, grade_to=13
    - 99_34_ocr_result.md   -> year=1999, grade_from=3, grade_to=4
    - 02_1113_ergaenzt_ocr_result.md -> year=2002, grade_from=11, grade_to=13
    - 19_910_ocr_result.md  -> year=2019, grade_from=9, grade_to=10
    """
    fn_re = re.compile(r"(\d{2})_(\d+)(?:_ergaenzt)?_ocr_result\.md")
    match = fn_re.match(filename)

    if not match:
        return None, None, None

    year_abbr_str = match.group(1)
    grades_str = match.group(2)

    # Parse year
    year_abbr = int(year_abbr_str)
    if year_abbr >= 90: # Covers 90-99
        year = 1900 + year_abbr
    else: # Covers 00-89 (e.g., 00 for 2000, 23 for 2023)
        year = 2000 + year_abbr

    # Parse grades
    grade_from, grade_to = None, None
    if len(grades_str) == 2: # e.g., "34" -> grades 3 and 4
        grade_from = int(grades_str[0])
        grade_to = int(grades_str[1])
    elif len(grades_str) == 3: # e.g., "910" -> grades 9 and 10
        grade_from = int(grades_str[0])
        grade_to = int(grades_str[1:])
    elif len(grades_str) == 4: # e.g., "1113" -> grades 11 and 13
        grade_from = int(grades_str[0:2])
        grade_to = int(grades_str[2:])
    else:
        logging.warning(f"Invalid grade format length in filename {filename}: '{grades_str}'")
        return None, None, None

    return year, grade_from, grade_to


def parse_exam_md(md_path: Path, file_year: int, file_grade_min: int, file_grade_max: int) -> dict:
    """
    Parse one exam markdown file into question dicts.
    Year, grade_min, and grade_max are provided from filename parsing.
    Returns: mapping id -> question_data (without answer)
    """
    text = md_path.read_text(encoding='utf-8')
    lines = text.splitlines()

    # Regex patterns
    points_section_re = re.compile(r'(\d)-Punkte[- ]Aufgaben', re.IGNORECASE)
    q_start_re = re.compile(r'^(?:A|B|C)?(\d+)[\.]?\s')
    choice_re = re.compile(r'^\(([A-E])\)\s*(.*)')

    current_points = None
    question_map = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Use search instead of match to find points section header anywhere in the line
        pm = points_section_re.search(line) # MODIFIED HERE
        if pm:
            current_points = int(pm.group(1))
            logging.debug(f"Found points section in {md_path.name}: {current_points} points. Line: '{line}'")
            i += 1
            continue

        qm = q_start_re.match(line)
        if qm:
            qnum = qm.group(1)
            stem_lines = [line[qm.end():].strip()]
            i += 1
            while i < len(lines) and \
                  not choice_re.match(lines[i]) and \
                  not q_start_re.match(lines[i]) and \
                  not points_section_re.search(lines[i]): # Also use search here for consistency during stem collection
                stem_lines.append(lines[i].strip())
                i += 1

            options = {}
            while i < len(lines):
                cm = choice_re.match(lines[i])
                if not cm:
                    break
                options[cm.group(1)] = cm.group(2).strip()
                i += 1
            
            if current_points is None:
                logging.warning(f"Question {qnum} in {md_path.name} found before any points section. Points will be None. Line: '{line[:50]}...'")


            qcode = f"{section_prefix(stem_lines)}{qnum}"
            qid = f"K{file_year}_{file_grade_min}-{file_grade_max}_{qcode}"
            question_map[qid] = {
                'id': qid,
                'year': file_year,
                'grade_min': file_grade_min,
                'grade_max': file_grade_max,
                'points': current_points,
                'question_type': 'multiple_choice' if options else 'free_response',
                'question': ' '.join(stem_lines),
                'options': options,
                'answer': None
            }
            logging.debug(f"Parsed question {qid} with points: {current_points} from {md_path.name}")
            continue

        i += 1
    return question_map


def section_prefix(stem_lines):
    """Infer section A/B/C from stem or context (stub)."""
    # For now, this remains a stub. A more sophisticated implementation
    # might inspect stem_lines or rely on question numbering if available.
    # Example: if question numbers are like A1, B1, C1.
    # The q_start_re `^(?:A|B|C)?(\d+)` already allows for this,
    # but the first group `(?:A|B|C)?` is non-capturing.
    # If q_start_re was `^([ABC])?(\d+)`, group 1 could be the section.
    return 'A'


def parse_solutions_md(md_path: Path) -> dict:
    """
    Parse solutions markdown into id->answer map.
    """
    year_re = re.compile(r'für den Känguru[- ]Mathematikwettbewerb\s+(\d{4})')
    grade_re = re.compile(r'Klassenstufe[n]?\s+(\d+)(?:\s*(?:bis|und)[\s-]*(\d+))?')
    table_re = re.compile(r'\|\s*Aufgabe')
    solution_map = {}

    lines = md_path.read_text(encoding='utf-8').splitlines()
    current_year = grade_min = grade_max = None
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        ym = year_re.search(l)
        if ym:
            current_year = int(ym.group(1))
            i += 1
            continue
        gm = grade_re.search(l)
        if gm:
            grade_min = int(gm.group(1))
            grade_max = int(gm.group(2)) if gm.group(2) else grade_min
            i += 1
            continue
        if table_re.match(l): # Match is okay here as table header should be at line start (after |)
            if current_year is None or grade_min is None or grade_max is None:
                logging.error(f"Skipping solution table at line {i+1} in {md_path.name} due to missing year or grade information.")
                i += 3
                continue
            header = [c.strip() for c in l.split('|')[1:-1]][1:]
            ans = [c.strip() for c in lines[i+2].split('|')[1:-1]][1:]
            if len(header) != len(ans):
                logging.error(f"Header and answer mismatch in solution table in {md_path.name} at line {i+1}. Header: {len(header)} items, Answers: {len(ans)} items.")
                i +=3
                continue
            for code, a in zip(header, ans):
                key = f"K{current_year}_{grade_min}-{grade_max}_{code}"
                if a in ('*', '–', ''):
                    solution_map[key] = None
                else:
                    letters = re.findall(r'[A-E]', a.upper()) # Convert to upper for consistency
                    if letters:
                        solution_map[key] = letters if len(letters) > 1 else letters[0]
                    else:
                        solution_map[key] = None # Or log a warning if answer format is unexpected
            i += 3
            continue
        i += 1
    return solution_map


def compute_difficulty(q):
    """
    Compute difficulty score [0,1] from grade, points, and length.
    Falls back to default values if required fields are missing.
    """
    if q.get('grade_min') is None or q.get('grade_max') is None: # Use .get for safer access
        logging.warning(f"Missing grade range for question {q.get('id', 'Unknown ID')}, using default avg_grade=8")
        avg_grade = 8
    else:
        avg_grade = (q['grade_min'] + q['grade_max']) / 2

    if q.get('points') is None:
        logging.warning(f"Missing points for question {q.get('id', 'Unknown ID')}, using default points=4 (average). Question will have difficulty based on this default.")
        points = 4 # Default points if not found
    else:
        points = q['points']

    g_norm = (avg_grade - 3) / (13 - 3)  # Assuming grades 3-13
    p_norm = (points - 3) / (5 - 3)    # Assuming points 3-5
    
    question_text = q.get('question', '')
    l_norm = math.log1p(len(question_text)) / math.log1p(250) # log1p for stability if len is 0
    
    raw = 0.6 * g_norm + 0.3 * p_norm + 0.1 * l_norm
    difficulty = 1 / (1 + math.exp(-12 * (raw - 0.5)))
    return difficulty


def main():
    parser = argparse.ArgumentParser(description='Process Känguru exams and solutions')
    parser.add_argument('input_dir', help='Directory containing exam and solution markdowns')
    parser.add_argument('--build-index', action='store_true')
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = OUTPUT_DIR # Already created at the top

    logging.info(f"Input directory: {inp.resolve()}")
    logging.info(f"Output directory: {out.resolve()}")
    logging.debug(f"Log file path: {log_file_path.resolve()}")

    all_md_files = list(inp.glob('*_ocr_result.md'))

    solutions_filename = 'kaenguru_loesungen_alle_ocr_result.md'
    sol_file = inp / solutions_filename

    exam_files = [f for f in all_md_files if f.name != solutions_filename]

    questions = {}
    # Use tqdm for progress bar over exam files
    for ef in tqdm(exam_files, desc="Processing Exam Files", unit="file"):
        logging.info(f"Processing exam file: {ef.name}")
        
        file_year, file_grade_min, file_grade_max = parse_filename_info(ef.name)
        
        if file_year is None or file_grade_min is None or file_grade_max is None:
            logging.warning(f"Could not parse required year/grade information from filename: {ef.name}. Skipping this file.")
            continue
        try:
            parsed_questions = parse_exam_md(ef, file_year, file_grade_min, file_grade_max)
            questions.update(parsed_questions)
        except Exception as e:
            logging.error(f"Failed to parse exam file {ef.name}: {e}", exc_info=True)


    solutions = {}
    if sol_file.is_file():
        logging.info(f"Parsing solutions file: {sol_file.name}")
        try:
            solutions = parse_solutions_md(sol_file)
        except Exception as e:
            logging.error(f"Failed to parse solutions file {sol_file.name}: {e}", exc_info=True)
    else:
        logging.warning(f"Solutions file not found: {sol_file}. Answers will not be populated.")

    records = []
    # Use tqdm for progress bar over processing questions for output/index
    for qid, q_data in tqdm(questions.items(), desc="Finalizing Questions", unit="question"):
        q_data['answer'] = solutions.get(qid) # qid should be consistent
        
        try:
            q_data['difficulty'] = compute_difficulty(q_data)
        except Exception as e:
            logging.error(f"Error computing difficulty for {qid}: {e}", exc_info=True)
            q_data['difficulty'] = None # Assign a default or None if computation fails
        
        year = q_data.get('year') # Use .get for safety
        if not isinstance(year, int):
            logging.error(f"Skipping question {qid} due to invalid year type: {type(year)}. Data: {q_data}")
            continue
            
        dest = out / str(year)
        dest.mkdir(exist_ok=True) # Parent 'out' dir already exists
        
        json_file_path = dest / f"{qid}.json"
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(q_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logging.error(f"Cannot write JSON for question {qid} to {json_file_path}: {e}", exc_info=True)
            continue # Skip adding to index if file cannot be written
            
        if args.build_index:
            rec = q_data.copy()
            rec['path_json'] = str(json_file_path.resolve()) # Use resolved path for index
            records.append(rec)

    if args.build_index and records:
        df = pd.DataFrame.from_records(records)
        
        # Always save index in the dataset directory
        index_output_path = OUTPUT_DIR / INDEX_FILENAME
        logging.info(f"Attempting to write index to: {index_output_path.resolve()}")
        
        try:
            df.to_parquet(index_output_path)
            logging.info(f"Index successfully written to {index_output_path.resolve()}")
        except Exception as e:
            logging.error(f"An error occurred while writing the index to {index_output_path.resolve()}: {e}", exc_info=True)
    elif args.build_index: # If --build-index was true but no records
        logging.info("No records were processed, so the index was not built.")


if __name__ == '__main__':
    main()