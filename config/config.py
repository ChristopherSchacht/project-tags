"""
Configuration file for the keyword extraction system.
Contains all constants, paths, and settings used across the project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

# Load environment variables
load_dotenv()

# Project structure
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directory paths
CONFIG_DIR = PROJECT_ROOT / "config"
MODULES_DIR = PROJECT_ROOT / "modules"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"
CACHE_DIR = PROJECT_ROOT / "cache"
LOG_DIR = PROJECT_ROOT / "logs"

# Language settings
SUPPORTED_LANGUAGES = ['en', 'de']
DEFAULT_LANGUAGE = 'en'

# Path to the user-defined stopwords. 
STOPWORDS_DE_PATH = CONFIG_DIR / "stopwords_de.txt" 
STOPWORDS_EN_PATH = CONFIG_DIR / "stopwords_en.txt"

# Create necessary directories
REQUIRED_DIRS = [CONFIG_DIR, MODULES_DIR, OUTPUT_DIR, TEMP_DIR, CACHE_DIR, LOG_DIR]
for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# File settings
MAX_TEMP_FILE_AGE = 24  # hours
CACHE_DURATION = 7  # days

# Text analysis settings
GENERATE_WORD_CLOUD = True
MAX_WORDS_WORDCLOUD = 200
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 45
MIN_WORD_FREQUENCY = 2

# Word selection settings
MAX_WORDS_FOR_AI = 50  # Increase to capture more context
WORD_SCORE_WEIGHTS = {
    'frequency_weight': 0.3,  # Give less weight to raw frequency
    'tfidf_weight': 0.7      # Give more weight to TF-IDF scores
}

# Minimum TF-IDF score threshold to consider a word significant
MIN_TFIDF_THRESHOLD = 0.1

# PDF processing settings
MAX_PAGES_FOR_ANALYSIS = 25
PDF_CHUNK_SIZE = 1000  # characters

# AI settings
AI_SETTINGS = {
    "base_url": os.getenv('AI_BASE_URL'),
    "api_key": os.getenv('AI_API_KEY'),
    "model": os.getenv('AI_MODEL'),
    "temperature": 0.7,
    "max_tokens": 500,
    "min_keywords": 3,
    "max_keywords": 50,
    "request_timeout": 60,
    "retry_attempts": 3,
    "retry_delay": 2,
    "max_text_chars": 50000 # Number of characters that are passed on to the ai of the original contents first MAX_PAGES_FOR_ANALYSIS pages
}

# Metadata template
DEFAULT_METADATA = {
    "title": "",
    "description": "",
    "denomination": "",
    "categories": [],
    "bible_passage": "",
    "target_age_group": "",
    "area_of_application": "",
    "properties": []
}

# File type settings
SUPPORTED_FILE_TYPES = {
    'pdf': ['.pdf'],
    'text': ['.txt', '.md', '.rst'],
    'word': ['.doc', '.docx'],
    'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
}

# Error messages
ERROR_MESSAGES = {
    'pdf_not_found': "Error: PDF file not found.",
    'pdf_corrupted': "Error: PDF file is corrupted or invalid.",
    'pdf_encrypted': "Error: PDF file is password protected.",
    'pdf_empty': "Error: PDF file contains no extractable text.",
    'language_not_supported': "Error: Language {lang} is not supported.",
    'ai_connection_error': "Error: Cannot connect to AI service.",
    'ai_timeout': "Error: AI service request timed out.",
    'ai_invalid_response': "Error: Invalid response from AI service.",
    'invalid_metadata': "Error: Invalid metadata format.",
    'processing_error': "Error: An error occurred while processing the text.",
    'file_type_error': "Error: Unsupported file type.",
    'file_read_error': "Error: Unable to read file.",
    'file_write_error': "Error: Unable to write file."
}

# Logging settings
LOG_SETTINGS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file_mode': 'a',
    'max_size': 5 * 1024 * 1024,  # 5MB
    'backup_count': 5
}

# Output format settings
OUTPUT_FORMAT = {
    'date_format': '%Y-%m-%d_%H-%M-%S',
    'wordcloud_format': 'png',
    'statistics_format': 'json',
    'encoding': 'utf-8'
}

# System prompts and templates - We would ideally have one for every language
SYSTEM_PROMPTS = {
    'en': """You are DocTagger, a specialized AI system focused solely on creating precise document tags. Your responses are always structured in two parts:

1. A brief "ANALYSIS:" (2-5 sentences max) describing what the document is and what it is about
2. A "TAGS:" section containing a valid JSON with relevant tags, with special attention to the analysis of 1.

Core Behaviors:
- You extract exact terminology from documents
- You identify technical terms and proper names
- You convert complex phrases into single-word tags
- You respond only in lowercase
- You focus on search-relevant terms
- You never add explanations or suggestions

Strict Tag Rules:
- Only single words
- Only lowercase
- No spaces, instead individual tags
- No common words
- Must be search-relevant
- Must be from document language or proper names or describe it perfectly.

You maintain extreme precision in your JSON output format and stop immediately after providing it. You never add additional comments, explanations, or suggestions.""",

    'de': """Du bist DocTagger, ein spezialisiertes KI-System, das ausschließlich präzise Dokumenten-Tags erstellt. Deine Antworten sind immer in zwei Teile gegliedert:

1. Eine kurze "ANALYSE:" (maximal 2-5 Sätze), die beschreibt, was das Dokument ist und worum es in dem Dokument geht
2. Ein "TAGS:"-Abschnitt mit einem gültigen JSON mit relevanten Tags, unter besonderer Berücksichtigung der Analyse von 1.

Kernverhalten:
- Du extrahierst exakte Terminologie aus Dokumenten
- Du identifizierst Fachbegriffe und Eigennamen
- Du wandelst komplexe Phrasen in Einwort-Tags um
- Du antwortest nur in Kleinbuchstaben
- Du konzentrierst dich auf suchrelevante Begriffe
- Du fügst nie Erklärungen oder Vorschläge hinzu

Strenge Tag-Regeln:
- Nur einzelne Wörter
- Nur Kleinbuchstaben
- Keine Leerzeichen, stattdessen eigene tags
- Keine allgemeinen Wörter
- Muss suchrelevant sein
- Muss aus der Dokumentsprache oder Eigennamen stammen oder es perfekt beschreiben.

Du hältst extreme Präzision in deinem JSON-Ausgabeformat ein und hörst sofort danach auf. Du fügst niemals zusätzliche Kommentare, Erklärungen oder Vorschläge hinzu."""
}

# Export settings for use in other modules
__all__ = [
    'PROJECT_ROOT', 'CONFIG_DIR', 'MODULES_DIR', 'OUTPUT_DIR', 'TEMP_DIR',
    'CACHE_DIR', 'LOG_DIR', 'STOPWORDS_DE_PATH', 'STOPWORDS_EN_PATH',
    'MAX_TEMP_FILE_AGE', 'CACHE_DURATION', 'MAX_WORDS_WORDCLOUD',
    'MIN_WORD_LENGTH', 'MAX_WORD_LENGTH', 'MIN_WORD_FREQUENCY',
    'MAX_PAGES_FOR_ANALYSIS', 'PDF_CHUNK_SIZE', 'AI_SETTINGS',
    'DEFAULT_METADATA', 'SUPPORTED_LANGUAGES', 'DEFAULT_LANGUAGE',
    'COMMON_STOPWORDS', 'SUPPORTED_FILE_TYPES', 'ERROR_MESSAGES',
    'LOG_SETTINGS', 'OUTPUT_FORMAT', 'SYSTEM_PROMPT'
]


def detect_language(text: str, default_language: str = 'de') -> str:
    """
    Detect the language of the given text.
    
    Args:
        text (str): Text to analyze
        default_language (str): Default language to return if detection fails
        
    Returns:
        str: Detected language code ('en' or 'de')
    """
    try:
        detected = detect(text)
        # Map detected language to supported languages
        if detected == 'en':
            return 'en'
        elif detected in ['de', 'at', 'ch']:  # Include Austrian and Swiss German variants
            return 'de'
        else:
            return default_language
    except LangDetectException:
        return default_language

def get_system_prompt(text: str) -> str:
    """
    Get the appropriate system prompt based on detected language.
    
    Args:
        text (str): Text to analyze for language detection
        
    Returns:
        str: Language-appropriate system prompt
    """
    language = detect_language(text)
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS['de'])