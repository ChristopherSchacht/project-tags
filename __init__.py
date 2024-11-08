# __init__.py (root level)
"""
Keyword extraction system root package.
"""
from modules.pdf_processor import PDFProcessor, PDFError
from modules.text_analyzer import TextAnalyzer, TextAnalysisError
from modules.ai_handler import AIHandler, AIError
from modules.utils import setup_logging, safe_file_write, get_timestamp

__version__ = '1.0.0'

__all__ = [
    'PDFProcessor',
    'TextAnalyzer',
    'AIHandler',
    'PDFError',
    'TextAnalysisError',
    'AIError',
    'setup_logging',
    'safe_file_write',
    'get_timestamp'
]