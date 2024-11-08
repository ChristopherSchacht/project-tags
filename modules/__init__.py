"""
Keyword extraction system module initialization.
"""
from pathlib import Path

# Import all components for easier access
from .pdf_processor import PDFProcessor, PDFError
from .text_analyzer import TextAnalyzer, TextAnalysisError
from .ai_handler import AIHandler, AIError
from .utils import setup_logging, safe_file_read, safe_file_write

__version__ = '1.0.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

# Export main components
__all__ = [
    'PDFProcessor',
    'TextAnalyzer',
    'AIHandler',
    'PDFError',
    'TextAnalysisError',
    'AIError',
    'setup_logging',
    'safe_file_read',
    'safe_file_write'
]

# Ensure all required directories exist
def _ensure_directories():
    """Create required directories if they don't exist."""
    from config.config import (
        OUTPUT_DIR,
        TEMP_DIR,
        CACHE_DIR,
        LOG_DIR
    )
    
    for directory in [OUTPUT_DIR, TEMP_DIR, CACHE_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories when module is imported
_ensure_directories()