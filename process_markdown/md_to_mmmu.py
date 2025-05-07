import argparse
import json
import re
from pathlib import Path
import pandas as pd
import logging
import os

# --- Configuration ---
OUTPUT_BASE_DIR_NAME = "dataset"
SOLUTIONS_FILENAME = "kaenguru_loesungen_alle_ocr_result.md"
IMAGE_DIR_RELATIVE_TO_MD_PARENT = "ocr_images" # Assumes ocr_images is sibling to the dir containing MD files
LOG_FILENAME = (Path(__file__).parent / "md_to_mmmu.log").resolve()

# Grade level normalization constants
GRADE_LEVEL_MIN_OVERALL = 3
GRADE_LEVEL_MAX_OVERALL = 13
GRADE_SPAN_OVERALL = GRADE_LEVEL_MAX_OVERALL - GRADE_LEVEL_MIN_OVERALL

# Point normalization constants
POINTS_MIN_3_5 = 3
POINTS_MAX_3_5 = 5
POINTS_SPAN_3_5 = POINTS_MAX_3_5 - POINTS_MIN_3_5

POINTS_MIN_6_10 = 6
POINTS_MAX_6_10 = 10
POINTS_SPAN_6_10 = POINTS_MAX_6_10 - POINTS_MIN_6_10

# --- Helper Functions ---

def parse_exam_filename(filename_str):
    """
    Parses year and grade levels from the exam filename.
    Example: "00_34_ocr_result.md" -> (2000, 3, 4, "3-4")
             "02_1113_ergaenzt_ocr_result.md" -> (2002, 11, 13, "11-13")
             "98_56_ocr_result.md" -> (1998, 5, 6, "5-6")
    """
    match = re.match(r"(\d{2})_(\d{2,4})(?:_ergaenzt)?_ocr_result\.md", filename_str)
    if not match:
        logging.warning(f"Could not parse filename: {filename_str}")
        return None, None, None, None

    year_short, grade_code = match.groups()

    year = int(year_short)
    if year < 50: # Assuming years 00-49 are 2000s, 50-99 are 1900s
        year += 2000
    else:
        year += 1900

    grade_min, grade_max = -1, -1
    grade_str_display = ""

    if len(grade_code) == 2: # e.g., "34"
        grade_min = int(grade_code[0])
        grade_max = int(grade_code[1])
        grade_str_display = f"{grade_min}-{grade_max}"
    elif len(grade_code) == 3: # e.g., "78", "910"
        if grade_code.startswith('9'): # "910"
             grade_min = 9
             grade_max = 10
        else:
            grade_min = int(grade_code[0])
            grade_max = int(grade_code[1:])
        grade_str_display = f"{grade_min}-{grade_max}"
    elif len(grade_code) == 4: # e.g., "1113"
        grade_min = int(grade_code[:2])
        grade_max = int(grade_code[2:])
        grade_str_display = f"{grade_min}-{grade_max}"
    else: # Fallback for single grade, e.g. if a file was "00_3_ocr_result.md"
        try:
            grade_min = int(grade_code)
            grade_max = int(grade_code)
            grade_str_display = str(grade_min)
        except ValueError:
            logging.warning(f"Could not parse grade_code '{grade_code}' from {filename_str}")
            return None, None, None, None


    if grade_min == -1 : # safety check
        logging.warning(f"Could not parse grade_code '{grade_code}' from {filename_str}")
        return None, None, None, None

    return year, grade_min, grade_max, grade_str_display

def normalize_grade_string_for_solutions(grade_header_str):
    """
    Normalizes grade strings from solution file headers to a consistent key format.
    e.g., "Klassenstufen 3 und 4" -> "3-4"
          "Klassenstufen 11 bis 13" -> "11-13"
    """
    match_und = re.search(r"(\d{1,2}) und (\d{1,2})", grade_header_str)
    if match_und:
        return f"{match_und.group(1)}-{match_und.group(2)}"
    match_bis = re.search(r"(\d{1,2}) bis (\d{1,2})", grade_header_str)
    if match_bis:
        return f"{match_bis.group(1)}-{match_bis.group(2)}"
    # Add more patterns if other formats exist
    logging.warning(f"Could not normalize grade string: {grade_header_str}")
    return grade_header_str # Fallback

def parse_solutions(solution_file_path):
    """
    Parses the main solution Markdown file.
    Returns a dictionary: solutions[year_str][grade_key_str][task_str] = answer_char
    """
    solutions_db = {}
    try:
        with open(solution_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"Solution file not found at {solution_file_path}")
        return solutions_db

    # Split content into sections per year
    # A year section starts with "## Lösungsbuchstaben" and ends before the next one or EOF
    year_block_regex = r"## Lösungsbuchstaben\s+für den Känguru-Mathematikwettbewerb\s*(\d{4})"
    # re.split with a capturing group results in: [before_first_delimiter, group1_val, after_delimiter_content, group1_val, ...]
    year_parts = re.split(year_block_regex, content)

    if len(year_parts) < 3: # Should have at least: content_before_first_year_header, first_year, first_year_content
        logging.warning("No year sections found in solutions file or format is unexpected.")
        return solutions_db

    # Iterate over pairs of (year_str, year_content_block)
    # The first element of year_parts is content before the first year header, skip it.
    current_year_str = None
    for i in range(1, len(year_parts)):
        part = year_parts[i]
        if i % 2 == 1: # This part is a year string (captured by \d{4})
            current_year_str = part.strip()
            solutions_db[current_year_str] = {}
            logging.debug(f"Initializing solutions for year: {current_year_str}")
        elif current_year_str: # This part is the content for the current_year_str
            year_content_block = part
            
            # Regex to find grade headers within the current year's content block
            # Captures "Klassenstufen X und Y" or "Klassenstufen X bis Y"
            # Also captures the "X und Y" or "X bis Y" part for normalization
            grade_header_regex = re.compile(
                r"^(?:#+\s*)?Klassenstufen\s+(\d{1,2}\s+(?:und|bis)\s+\d{1,2})", 
                re.MULTILINE | re.IGNORECASE
            )
            
            grade_header_matches = list(grade_header_regex.finditer(year_content_block))
            
            if not grade_header_matches:
                logging.warning(f"No grade headers found for year {current_year_str} in block: {year_content_block[:200]}...") # Log start of block

            for idx, match_obj in enumerate(grade_header_matches):
                grade_description_part = match_obj.group(1)  # e.g. "3 und 4" or "11 bis 13"
                
                # Prepend "Klassenstufen" to match expected format for normalization
                normalized_grade_key = normalize_grade_string_for_solutions(f"Klassenstufen {grade_description_part}")

                if normalized_grade_key == f"Klassenstufen {grade_description_part}": # Normalization failed, log and skip
                    logging.warning(f"Could not normalize grade header '{grade_description_part}' for year {current_year_str}. Skipping this grade section.")
                    continue
                
                solutions_db[current_year_str][normalized_grade_key] = {}
                
                # Determine the content for this grade section
                content_start_pos = match_obj.end()
                content_end_pos = grade_header_matches[idx+1].start() if idx + 1 < len(grade_header_matches) else len(year_content_block)
                grade_section_text = year_content_block[content_start_pos:content_end_pos]
                
                task_lines_buffer = []
                answer_lines_buffer = []

                for line in grade_section_text.strip().split('\n'):
                    line_s = line.strip()
                    if not line_s: continue # Skip empty lines
                    if line_s.startswith('| Aufgabe |'):
                        task_lines_buffer.append(line_s)
                    elif line_s.startswith('| Antwort |'):
                        answer_lines_buffer.append(line_s)
                    elif line_s.startswith('| :-- |'): # Skip markdown table separator lines
                        continue
                
                if len(task_lines_buffer) != len(answer_lines_buffer):
                    logging.warning(f"Mismatch in 'Aufgabe' ({len(task_lines_buffer)}) and 'Antwort' ({len(answer_lines_buffer)}) table rows for year {current_year_str}, grade {normalized_grade_key}.")
                    # Continue with the minimum of the two, to salvage some data
                
                for tbl_idx in range(min(len(task_lines_buffer), len(answer_lines_buffer))):
                    header_line = task_lines_buffer[tbl_idx]
                    answer_line = answer_lines_buffer[tbl_idx]
                    
                    raw_tasks = [h.strip() for h in header_line.split('|')[2:] if h.strip()] # Skip first two empty items from split
                    raw_answers = [a.strip() for a in answer_line.split('|')[2:] if a.strip()] # Same here
                    
                    if len(raw_tasks) == len(raw_answers):
                        for task, ans in zip(raw_tasks, raw_answers):
                            # Ensure task key is not empty, as it's used as a dict key
                            if task: 
                                solutions_db[current_year_str][normalized_grade_key][task] = ans
                            else:
                                logging.warning(f"Empty task key found for year {current_year_str}, grade {normalized_grade_key}. Line: {header_line}")
                    else:
                        logging.warning(f"Mismatch in task/answer count within a table for year {current_year_str}, grade {normalized_grade_key}: Tasks '{raw_tasks}' vs Answers '{raw_answers}' in lines:\n{header_line}\n{answer_line}")
            
            if not solutions_db[current_year_str] and grade_header_matches: # If headers were found but no data populated
                 logging.warning(f"Data for grade sections might be empty or not parsed correctly for year {current_year_str}, even though grade headers were found.")
            elif not grade_header_matches and year_content_block.strip(): # If there was content but no headers
                 logging.warning(f"No grade headers like 'Klassenstufen X und Y' found in content for year {current_year_str}.")
            current_year_str = None # Reset for the next block

    return solutions_db


def calculate_difficulty(avg_grade, points, current_min_points, current_max_points):
    """Calculates a difficulty score between 0 and 1."""
    if GRADE_SPAN_OVERALL == 0: # Avoid division by zero
        norm_grade = 0.5
    else:
        norm_grade = (avg_grade - GRADE_LEVEL_MIN_OVERALL) / GRADE_SPAN_OVERALL
    
    current_points_span = current_max_points - current_min_points
    if current_points_span == 0: # Avoid division by zero
        norm_points = 0.5
    else:
        norm_points = (points - current_min_points) / current_points_span
        
    difficulty_score = (norm_grade * 0.6) + (norm_points * 0.4)
    return max(0.0, min(1.0, difficulty_score)) # Clamp between 0 and 1

# --- Helper: Extract question key robustly ---
def extract_question_key(raw_key):
    key = str(raw_key).strip()
    key = re.sub(r'^[\(\[]*', '', key)  # Remove leading ( or [
    key = re.sub(r'[\)\]:\.]?$', '', key)  # Remove trailing ) or : or .
    key = key.replace(' ', '')
    return key

option_regex = re.compile(r"^\s*(?:\(\s*([A-E])\s*\)|([A-E])\s*:)\s*(.*)")
image_regex = re.compile(r"!\[.*?\]\((.*?)\)") # Capture path from ![alt](path)

def parse_exam_file(exam_file_path, year, grade_min, grade_max, grade_str_key_solutions, solutions_db, input_dir_path, year_output_dir):
    """
    Parses a single exam Markdown file and extracts questions.
    `grade_str_key_solutions` is the key like "3-4" used in solutions_db.
    """
    questions_data = []
    
    try:
        with open(exam_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Exam file not found: {exam_file_path}")
        return questions_data

    current_points = None
    current_min_cat_points = None # For 3/4/5 or 6/8/10 system
    current_max_cat_points = None

    question_text_buffer = []
    options_buffer = []
    image_paths_buffer = []
    current_question_number_raw = None
    
    # Matches # or ##, optional dash, optional 'Punkte', optional dash, Aufgaben/Fragen, with or without dashes/spaces
    points_header_regex = re.compile(r"^#{1,3}\s*(\d+)\s*-?\s*Punkte\s*-?\s*(Aufgaben|Fragen)", re.IGNORECASE)
    # Matches A1. Text, A1) Text, (A1) Text, A1 Text, A1: Text, etc.
    question_start_regex = re.compile(r"^\s*(?:\(?([A-C]?[0-9]{1,2})\)?[\.:\)]?\s+)(.*)")
    option_regex = re.compile(r"^\s*(?:\(\s*([A-E])\s*\)|([A-E])\s*:)\s*(.*)")
    image_regex = re.compile(r"!\[.*?\]\((.*?)\)") # Capture path from ![alt](path)

    parsing_questions_started = False
    question_keys_in_section = []

    # Regex to find all individual options like "(A) text" within a line
    multi_option_finder_regex = re.compile(r"\(\s*([A-E])\s*\)\s*(.*?)(?=\s*\(\s*[A-E]\s*\)\s*|$)")

    for line_num, line in enumerate(lines):
        line_strip = line.strip()

        # Check for points header
        points_match = points_header_regex.match(line_strip)
        if points_match:
            if current_question_number_raw and question_text_buffer: # Save previous question
                questions_data.append(finalize_question(
                    year, grade_min, grade_max, grade_str_key_solutions, solutions_db,
                    extract_question_key(current_question_number_raw), current_points,
                    question_text_buffer, options_buffer, image_paths_buffer,
                    current_min_cat_points, current_max_cat_points, exam_file_path, input_dir_path, year_output_dir
                ))
                question_keys_in_section.append(extract_question_key(current_question_number_raw))
                question_text_buffer, options_buffer, image_paths_buffer, current_question_number_raw = [], [], [], None

            current_points = int(points_match.group(1))
            parsing_questions_started = True
            # Determine point system for difficulty calculation
            if current_points in [3,4,5]:
                current_min_cat_points = POINTS_MIN_3_5
                current_max_cat_points = POINTS_MAX_3_5
            elif current_points in [6,8,10]:
                current_min_cat_points = POINTS_MIN_6_10
                current_max_cat_points = POINTS_MAX_6_10
            else: # Fallback or warning
                logging.warning(f"Unknown point category {current_points} in {exam_file_path}. Defaulting difficulty points.")
                current_min_cat_points = current_points 
                current_max_cat_points = current_points
            continue

        if not parsing_questions_started:
            continue # Skip intro lines

        # Check for question start
        q_match = question_start_regex.match(line_strip)
        if q_match:
            if current_question_number_raw and question_text_buffer: # Save previous question
                questions_data.append(finalize_question(
                    year, grade_min, grade_max, grade_str_key_solutions, solutions_db,
                    extract_question_key(current_question_number_raw), current_points,
                    question_text_buffer, options_buffer, image_paths_buffer,
                    current_min_cat_points, current_max_cat_points, exam_file_path, input_dir_path, year_output_dir
                ))
                question_keys_in_section.append(extract_question_key(current_question_number_raw))
                question_text_buffer, options_buffer, image_paths_buffer, current_question_number_raw = [], [], [], None
            current_question_number_raw = q_match.group(1)
            question_text_buffer = [q_match.group(2).strip()]
            options_buffer = []
            image_paths_buffer = []
            for img_match in image_regex.finditer(q_match.group(2)):
                image_paths_buffer.append(img_match.group(1))
            continue

        if current_question_number_raw: # If we are inside a question block
            # Check for options
            opt_match = option_regex.match(line_strip)
            
            if opt_match:
                found_options_on_line = multi_option_finder_regex.findall(line_strip)
                
                if found_options_on_line:
                    for opt_letter, opt_text_content in found_options_on_line:
                        formatted_option = f"({opt_letter.strip()}) {opt_text_content.strip()}"
                        options_buffer.append(formatted_option)
                        # also parse images from this specific option's text
                        for img_match in image_regex.finditer(opt_text_content):
                            image_paths_buffer.append(img_match.group(1))
                else:
                    # No "(X) text" options found - fallback to original opt_match for "A: text" style
                    option_letter_orig = opt_match.group(1) or opt_match.group(2) # group1 for (X), group2 for X:
                    option_text_orig = opt_match.group(3).strip()
                    
                    # Store in the canonical "(A) Text" format
                    options_buffer.append(f"({option_letter_orig.strip()}) {option_text_orig}")
                    for img_match in image_regex.finditer(option_text_orig):
                        image_paths_buffer.append(img_match.group(1))

                    # Fallback for single-line options like 'A: ... B: ...'
                    if len(options_buffer) == 1:
                        all_alt_options = re.findall(r'([A-E]):\s*([^A-E]*)', option_text_orig)
                        if len(all_alt_options) >= 2:
                            # Remove the previously added single entry
                            options_buffer.pop()
                            # Split line into individual options
                            split_pattern = re.compile(r'([A-E]):\s*([^A-E]*)')
                            matches = list(split_pattern.finditer(line_strip))
                            for idx, match in enumerate(matches):
                                letter = match.group(1)
                                start = match.end()
                                end = matches[idx+1].start() if idx+1 < len(matches) else len(line_strip)
                                text = line_strip[start:end].strip()
                                # Prepend the matched text (in case the value is not empty)
                                full_option = f"({letter}) {match.group(2).strip()} {text}".strip()
                                options_buffer.append(full_option)
                                for img_match in image_regex.finditer(full_option):
                                    image_paths_buffer.append(img_match.group(1))
                
                # Since this line was processed as an option line, continue to the next line in the input file.
                continue 

            elif question_text_buffer: # Continue collecting question text if not an option line
                # Only add if it's not an empty line or just an image line that's already processed by image_regex
                if line_strip:
                     question_text_buffer.append(line_strip)
                for img_match in image_regex.finditer(line_strip):
                    image_paths_buffer.append(img_match.group(1))
    
    # Add the last question after loop finishes
    if current_question_number_raw and question_text_buffer:
        questions_data.append(finalize_question(
            year, grade_min, grade_max, grade_str_key_solutions, solutions_db,
            extract_question_key(current_question_number_raw), current_points,
            question_text_buffer, options_buffer, image_paths_buffer,
            current_min_cat_points, current_max_cat_points, exam_file_path, input_dir_path, year_output_dir
        ))
        question_keys_in_section.append(extract_question_key(current_question_number_raw))
    
    # Log if first question is not A1 or 1
    if question_keys_in_section and not (question_keys_in_section[0] in ['A1', '1']):
        logging.warning(f"First question key is not A1 or 1: {question_keys_in_section[0]} in {exam_file_path}")
    
    return [q for q in questions_data if q is not None] # Filter out None if finalize_question returns it

def finalize_question(year, grade_min, grade_max, grade_str_key_solutions, solutions_db,
                      q_num_raw, points_val, q_text_buf, opts_buf, img_paths_buf,
                      min_pts_cat, max_pts_cat, md_file_path, input_dir_path_obj, year_output_dir):
    """Helper to construct the question dictionary and add it to the list."""
    if not q_text_buf or not opts_buf or points_val is None:
        logging.debug(f"Skipping question finalization due to missing text, options, or points: q_num_raw={q_num_raw}, md_file_path={md_file_path}")
        return None

    full_question_text = " ".join(q_text_buf).strip()
    md_file_dir = Path(md_file_path).parent
    # Define a common base path for resolving image paths, which is the parent of the markdown directory
    # e.g., if md_file_dir is ".../ocr_output/ocr_markdown", base_path_for_images is ".../ocr_output"
    base_path_for_images = md_file_dir.parent

    # Create list of unique, ordered, relative image paths for the JSON output
    unique_ordered_rel_image_paths_for_json = []
    seen_rel_paths_set = set()

    for img_path_from_md_scan in img_paths_buf: # These are paths like '../ocr_images/...' relative to MD file
        try:
            abs_path_obj = (md_file_dir / img_path_from_md_scan).resolve()
            rel_path_str = os.path.relpath(abs_path_obj, year_output_dir)
            if rel_path_str not in seen_rel_paths_set:
                unique_ordered_rel_image_paths_for_json.append(rel_path_str)
                seen_rel_paths_set.add(rel_path_str)
        except Exception as e:
            logging.warning(f"Error resolving or storing image path '{img_path_from_md_scan}' from {md_file_path} relative to {year_output_dir}: {e}. Skipping this image path.")
            # Continue to process other images

    # Check if images belong to options rather than the question itself
    images_are_options = False
    
    # Check if each option in opts_buf is just a label with no content (e.g., "(A) ", "(B) ")
    empty_options = all(opt.strip().endswith(")") or opt.strip().endswith(") ") for opt in opts_buf)
    
    # If we have empty options and the same number of images as options, assume the images belong to the options
    if empty_options and (len(unique_ordered_rel_image_paths_for_json) >= len(opts_buf) - 1):
        images_are_options = True

    # Replace MD image tags with placeholders, or remove if they belong to options
    def replace_md_image_with_placeholder_tag(match, context="question"):
        markdown_relative_path = match.group(1) # e.g., '../ocr_images/img.jpg'
        try:
            abs_path_from_text = (md_file_dir / markdown_relative_path).resolve()
            key_for_lookup = os.path.relpath(abs_path_from_text, year_output_dir)
            if key_for_lookup in unique_ordered_rel_image_paths_for_json:
                image_index = unique_ordered_rel_image_paths_for_json.index(key_for_lookup) + 1
                # For question, include image placeholder only if images are not part of options
                if context == "question" and images_are_options:
                    return "" # Remove image from question if it belongs to an option
                return f"<image {image_index}>"
            else:
                logging.warning(f"Image '{markdown_relative_path}' (resolved: {key_for_lookup}) from {context} text '{q_num_raw}' in {md_file_path} not found in the collected image list. Original tag kept.")
                return match.group(0) # Keep original markdown tag
        except Exception as e:
            logging.error(f"Error during image tag replacement for '{markdown_relative_path}' in {md_file_path}: {e}. Original tag kept.")
            return match.group(0) # Keep original markdown tag on error

    # Replace image tags in question text
    modified_full_question_text = image_regex.sub(lambda m: replace_md_image_with_placeholder_tag(m, "question"), full_question_text)
    modified_full_question_text = modified_full_question_text.strip()
    
    # Process options if necessary
    modified_opts_buf = []
    if images_are_options:
        # Assign each image to its respective option
        for i, opt in enumerate(opts_buf):
            if i < len(unique_ordered_rel_image_paths_for_json):
                image_index = i + 1  # 1-based index
                modified_opts_buf.append(f"{opt.strip()} <image {image_index}>")
            else:
                modified_opts_buf.append(opt)
    else:
        modified_opts_buf = opts_buf
    
    year_str = str(year).strip()
    solution_char = "N/A" # Default if no solution found

    if year_str in solutions_db:
        if grade_str_key_solutions in solutions_db[year_str]:
            current_solutions_for_grade = solutions_db[year_str][grade_str_key_solutions]
            solution_keys_from_db = list(current_solutions_for_grade.keys())

            # Attempt 1: Direct match with q_num_raw (e.g., "A1" from exam matches "A1" in solutions)
            if q_num_raw in current_solutions_for_grade:
                solution_char = current_solutions_for_grade[q_num_raw]
            else:
                # Attempt 2: Sort keys like A1, A2, B1, B2 or 1, 2
                def natural_sort_key_for_solutions(key_str):
                    match = re.match(r'([ABCabc])?(\d+)', key_str)
                    if match:
                        letter_part, num_part = match.groups()
                        if letter_part: # A, B, or C
                            letter_val = {'a': 0, 'b': 1, 'c': 2}.get(letter_part.lower(), 3)
                            return (letter_val, int(num_part))
                        else: # Just a number
                            return (4, int(num_part)) # Numeric keys after C, or if no A/B/C sections
                    return (5, key_str) # Fallback for completely unexpected keys (sort last)

                sorted_solution_keys = sorted(solution_keys_from_db, key=natural_sort_key_for_solutions)
                
                fallback_key_used = None
                # Handle numeric q_num_raw (e.g. "5") mapping to solution keys
                if q_num_raw.isdigit():
                    try:
                        # Map numeric q_num_raw to corresponding solution key by index
                        target_idx = int(q_num_raw) - 1 
                        if 0 <= target_idx < len(sorted_solution_keys):
                            potential_fallback_key = sorted_solution_keys[target_idx]
                            solution_char = current_solutions_for_grade[potential_fallback_key]
                            fallback_key_used = potential_fallback_key
                            logging.info(
                                f"Solution for numeric task '{q_num_raw}' (exam) found using fallback index {target_idx} "
                                f"(solution key: '{potential_fallback_key}') in {md_file_path} for {year_str}/{grade_str_key_solutions}."
                            )
                    except ValueError:
                        logging.error(f"ValueError converting q_num_raw '{q_num_raw}' to int after isdigit check. File: {md_file_path}")
                        pass 
                
                if fallback_key_used is None:
                    logging.warning(
                        f"Solution not found for Task '{q_num_raw}' (exam key) for Year {year_str}, Grade {grade_str_key_solutions} in {md_file_path}."
                    )
                    if not solution_keys_from_db:
                        logging.warning(f"  No solution keys available in solutions_db for {year_str}/{grade_str_key_solutions}.")
                    else:
                        logging.warning(f"  Exam task key: '{q_num_raw}'. Available solution keys from DB: {solution_keys_from_db}")
                        logging.warning(f"  Attempted sorted solution keys for fallback: {sorted_solution_keys}")
        else: # Grade key not found for the year
            logging.warning(f"Grade key '{grade_str_key_solutions}' not found for Year {year_str} in solutions_db (for file {md_file_path}).")
            if year_str in solutions_db:
                 logging.warning(f"  Available grade keys for Year {year_str}: {list(solutions_db[year_str].keys())}")
            else:
                 logging.warning(f"  Year {year_str} has no entries in solutions_db.")
    else: # Year not found in solutions_db
        logging.warning(f"Year '{year_str}' not found in solutions_db (for file {md_file_path}).")
        logging.warning(f"  Available years in solutions_db: {list(solutions_db.keys())}")

    avg_grade = (grade_min + grade_max) / 2.0
    difficulty = calculate_difficulty(avg_grade, points_val, min_pts_cat, max_pts_cat)
    mmmu_id = f"MathKangaroo_{year_str}_{grade_min}-{grade_max}_{q_num_raw.replace('.', '')}"
    return {
        "id": mmmu_id,
        "question_type": "multiple-choice",
        "question": modified_full_question_text,
        "options": modified_opts_buf,
        "answer": solution_char,
        "image_paths": unique_ordered_rel_image_paths_for_json,
        "grade_level_raw": f"{grade_min}-{grade_max}",
        "grade_level_min": grade_min,
        "grade_level_max": grade_max,
        "year": year,
        "points": points_val,
        "question_number_raw": q_num_raw,
        "question_difficulty": difficulty
    }

def main():
    parser = argparse.ArgumentParser(description="Parse Math Kangaroo exam papers to MMMU format.")
    parser.add_argument("input_dir", help="Directory path containing the Markdown exam files and the solution file.")
    parser.add_argument("--build-index", action="store_true", help="Save the built dataset as a single Parquet file.")
    args = parser.parse_args()

    input_dir_path = Path(args.input_dir).resolve()
    script_dir = Path(__file__).parent.resolve()
    output_dir_base = (script_dir / ".." / OUTPUT_BASE_DIR_NAME).resolve()

    if not input_dir_path.is_dir():
        logging.error(f"Input directory '{args.input_dir}' not found or not a directory.")
        return

    solution_file_path = input_dir_path / SOLUTIONS_FILENAME
    if not solution_file_path.exists():
        logging.error(f"Solution file '{SOLUTIONS_FILENAME}' not found in '{args.input_dir}'.")
        return

    logging.info("Parsing solutions...")
    solutions_db = parse_solutions(solution_file_path)
    if not solutions_db:
        logging.error("Failed to parse solutions. Exiting.")
        return
    logging.debug(f"Solutions DB loaded for years: {list(solutions_db.keys())}")

    all_questions_for_parquet = []

    logging.info("Processing exam files...")
    for md_file in input_dir_path.glob("*_ocr_result.md"):
        if md_file.name == SOLUTIONS_FILENAME:
            continue

        logging.info(f"  Parsing {md_file.name}...")
        year, grade_min, grade_max, grade_str_display = parse_exam_filename(md_file.name)

        if year is None:
            logging.warning(f"    Skipping {md_file.name} due to filename parsing error.")
            continue
        
        grade_key_for_solutions = f"{grade_min}-{grade_max}"

        # Create year-specific output directory
        year_str = str(year).strip()
        year_output_dir = output_dir_base / year_str
        year_output_dir.mkdir(parents=True, exist_ok=True)

        questions = parse_exam_file(md_file, year, grade_min, grade_max, grade_key_for_solutions, solutions_db, input_dir_path, year_output_dir)

        if not questions:
            logging.warning(f"    No questions extracted from {md_file.name}.")
            continue

        for q_data in questions:
            if q_data is None: continue
            # Generate filename for individual JSON
            # id: MathKangaroo_2000_3-4_A1
            json_filename = f"{q_data['id']}.json"
            json_file_path = year_output_dir / json_filename
            
            try:
                with open(json_file_path, 'w', encoding='utf-8') as jf:
                    json.dump(q_data, jf, ensure_ascii=False, indent=4)
            except IOError as e:
                logging.error(f"Error writing JSON file {json_file_path}: {e}")


            if args.build_index:
                all_questions_for_parquet.append(q_data)
        
        logging.info(f"    Successfully processed {md_file.name}, extracted {len(questions)} questions.")

    if args.build_index:
        if all_questions_for_parquet:
            logging.info(f"\nBuilding Parquet index for {len(all_questions_for_parquet)} questions...")
            df = pd.DataFrame(all_questions_for_parquet)
            parquet_output_dir = output_dir_base
            parquet_output_dir.mkdir(parents=True, exist_ok=True)
            parquet_file_path = parquet_output_dir / "math_kangaroo_dataset.parquet"
            try:
                df.to_parquet(parquet_file_path, index=False)
                logging.info(f"Successfully saved Parquet index to {parquet_file_path}")
            except Exception as e:
                logging.error(f"Error saving Parquet file: {e}")
                logging.error("Please ensure you have 'pyarrow' and 'pandas' installed ('pip install pandas pyarrow').")
        else:
            logging.warning("\nNo questions collected to build Parquet index.")
            
    logging.info("\nProcessing complete.")

if __name__ == "__main__":
    # Set up logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_handler = logging.FileHandler(str(LOG_FILENAME), mode='w', encoding='utf-8')
    log_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    log_handler.setFormatter(formatter)
    logging.getLogger().addHandler(log_handler)
    logging.getLogger().setLevel(logging.DEBUG)
    main()
