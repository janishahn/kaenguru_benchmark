"""
Shared logging setup based on project configuration.
"""
import logging
from pathlib import Path
import yaml

def setup_logging_from_config(config_path: Path) -> None:
    """
    Set up global logging based on the config.yaml settings.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = yaml.safe_load(config_path.read_text())

    base_dir = Path(config["processing_output_base_dir"])
    log_filename = config["logging"]["pipeline_log_filename"]
    level_str = config["logging"].get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    # Ensure output directory exists
    base_dir.mkdir(parents=True, exist_ok=True)
    log_file = base_dir / log_filename

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler for pipeline log
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler for warnings and above
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
