"""
Shared utility functions for the keyword extraction system.
"""
import logging
import os
from pathlib import Path
from typing import Optional, Union
import json
from datetime import datetime

from config import OUTPUT_DIR, TEMP_DIR

def setup_logging(name: str) -> logging.Logger:
    """
    Set up logging for a module.
    
    Args:
        name: Module name
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = OUTPUT_DIR / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        logger.addHandler(console_handler)
    
    return logger

def safe_file_write(data: Union[str, dict], 
                    filepath: Union[str, Path], 
                    ensure_dir: bool = True) -> Path:
    """
    Safely write data to file with directory creation.
    
    Args:
        data: Data to write
        filepath: Output file path
        ensure_dir: Create directory if needed
        
    Returns:
        Path: Path to written file
    """
    filepath = Path(filepath)
    
    if ensure_dir:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    temp_file = TEMP_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        if isinstance(data, dict):
            temp_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            temp_file.write_text(str(data))
            
        # Move to final location
        temp_file.replace(filepath)
        return filepath
        
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise IOError(f"Failed to write file: {str(e)}")

def safe_file_read(filepath: Union[str, Path], as_json: bool = False) -> Union[str, dict]:
    """
    Safely read file with error handling.
    
    Args:
        filepath: File to read
        as_json: Parse as JSON
        
    Returns:
        Union[str, dict]: File contents
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
        
    try:
        content = filepath.read_text(encoding='utf-8')
        return json.loads(content) if as_json else content
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in file: {filepath}")
    except Exception as e:
        raise IOError(f"Failed to read file: {str(e)}")

def clean_filename(filename: str) -> str:
    """
    Create safe filename from string.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Cleaned filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Ensure reasonable length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
        
    return filename

def get_timestamp() -> str:
    """Get formatted timestamp for filenames."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def create_temp_path(prefix: str = "", suffix: str = "") -> Path:
    """
    Create temporary file path.
    
    Args:
        prefix: Filename prefix
        suffix: Filename suffix
        
    Returns:
        Path: Temporary file path
    """
    timestamp = get_timestamp()
    filename = f"{prefix}_{timestamp}{suffix}" if prefix else f"temp_{timestamp}{suffix}"
    return TEMP_DIR / clean_filename(filename)

# Initialize required directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Clean up old temp files on import
def cleanup_temp_files(max_age_hours: int = 24):
    """Remove old temporary files."""
    try:
        current_time = datetime.now()
        for temp_file in TEMP_DIR.glob("temp_*"):
            file_age = current_time - datetime.fromtimestamp(temp_file.stat().st_mtime)
            if file_age.total_seconds() > max_age_hours * 3600:
                temp_file.unlink()
    except Exception as e:
        logger = setup_logging("utils")
        logger.warning(f"Failed to cleanup temp files: {str(e)}")

cleanup_temp_files()
