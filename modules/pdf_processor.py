"""
PDF processing module with enhanced error handling and project structure integration.
"""
from pathlib import Path
from typing import Dict, Union, Tuple
import PyPDF2
from PyPDF2.errors import PdfReadError
from langdetect import detect, LangDetectException
import logging
import os

# Project imports using absolute paths
from config.config import (
    MAX_PAGES_FOR_ANALYSIS,
    PDF_CHUNK_SIZE,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    ERROR_MESSAGES,
    SUPPORTED_FILE_TYPES,
    TEMP_DIR,
    LOG_SETTINGS
)

# Set up logging based on config
logging.basicConfig(
    level=getattr(logging, LOG_SETTINGS['level']),
    format=LOG_SETTINGS['format'],
    datefmt=LOG_SETTINGS['date_format']
)
logger = logging.getLogger(__name__)

class PDFError(Exception):
    """Base class for PDF processing errors."""
    pass

class PDFEncryptedError(PDFError):
    """Error for encrypted/password-protected PDFs."""
    pass

class PDFCorruptedError(PDFError):
    """Error for corrupted PDF files."""
    pass

class PDFEmptyError(PDFError):
    """Error for PDFs with no extractable text."""
    pass

class PDFProcessor:
    """Handles PDF processing with enhanced error handling."""
    
    def __init__(self, pdf_path: Union[str, Path]):
        """
        Initialize PDF processor.
        
        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = Path(pdf_path)
        self.text = ""
        self.language = DEFAULT_LANGUAGE
        self.stats = {
            'total_pages': 0,
            'processed_pages': 0,
            'total_characters': 0,
            'detected_language': None,
            'success': False,
            'error': None,
            'warnings': []
        }
        
        # Create temp directory for processing
        self.temp_dir = TEMP_DIR / "pdf_processing"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def validate_pdf(self) -> bool:
        """
        Validate PDF file with detailed error checking.
        
        Returns:
            bool: True if valid
            
        Raises:
            Various PDF exceptions based on error type
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(ERROR_MESSAGES['pdf_not_found'])

        if self.pdf_path.suffix.lower() not in SUPPORTED_FILE_TYPES['pdf']:
            raise ValueError(ERROR_MESSAGES['file_type_error'])

        # Check file permissions
        if not os.access(self.pdf_path, os.R_OK):
            raise PermissionError(ERROR_MESSAGES['file_read_error'])

        try:
            with open(self.pdf_path, 'rb') as file:
                try:
                    reader = PyPDF2.PdfReader(file)
                    
                    # First try decrypting with empty password if needed
                    if reader.is_encrypted:
                        if not reader.decrypt(''):
                            raise PDFEncryptedError(ERROR_MESSAGES['pdf_encrypted'])
                    
                    # Verify we can read pages
                    if len(reader.pages) == 0:
                        raise PDFEmptyError(ERROR_MESSAGES['pdf_empty'])
                        
                    # Try to access first page to verify readability
                    _ = reader.pages[0].extract_text()
                    
                    return True

                except PdfReadError as e:
                    # More specific error handling based on the error message
                    error_msg = str(e).lower()
                    if "password" in error_msg or "encrypted" in error_msg:
                        raise PDFEncryptedError(ERROR_MESSAGES['pdf_encrypted'])
                    raise PDFCorruptedError(f"{ERROR_MESSAGES['pdf_corrupted']} Details: {str(e)}")
                    
        except (PDFError, FileNotFoundError, ValueError, PermissionError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during PDF validation: {str(e)}")
            raise PDFCorruptedError(f"{ERROR_MESSAGES['pdf_corrupted']} Details: {str(e)}")

    def extract_text(self) -> Tuple[str, Dict]:
        """
        Extract text with comprehensive error handling.
        
        Returns:
            Tuple[str, Dict]: Processed text and statistics
            
        Raises:
            Various PDF exceptions based on error type
        """
        try:
            logger.info(f"Starting text extraction from {self.pdf_path}")
            self.validate_pdf()
            
            with open(self.pdf_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    # Handle encryption if needed
                    if pdf_reader.is_encrypted:
                        if not pdf_reader.decrypt(''):
                            raise PDFEncryptedError(ERROR_MESSAGES['pdf_encrypted'])
                    
                    self.stats['total_pages'] = len(pdf_reader.pages)
                    pages_to_process = min(MAX_PAGES_FOR_ANALYSIS, len(pdf_reader.pages))
                    
                    logger.info(f"Processing {pages_to_process} pages out of {self.stats['total_pages']}")
                    
                    extracted_text = []
                    page_errors = []
                    
                    for page_num in range(pages_to_process):
                        try:
                            logger.debug(f"Processing page {page_num + 1}")
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            
                            if not page_text.strip():
                                warning = f"Page {page_num + 1} has no extractable text"
                                logger.warning(warning)
                                page_errors.append(warning)
                                continue
                                
                            cleaned_text = self.clean_text(page_text)
                            if cleaned_text:
                                temp_file = self.temp_dir / f"page_{page_num}.txt"
                                try:
                                    with open(temp_file, 'w', encoding='utf-8') as f:
                                        f.write(cleaned_text)
                                except Exception as e:
                                    logger.error(f"Failed to write temporary file: {str(e)}")
                                
                                extracted_text.append(cleaned_text)
                                self.stats['processed_pages'] += 1
                            else:
                                warning = f"Page {page_num + 1} yielded no text after cleaning"
                                logger.warning(warning)
                                page_errors.append(warning)
                                
                        except Exception as e:
                            error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                            logger.error(error_msg)
                            page_errors.append(error_msg)
                            continue

                    if not extracted_text:
                        raise PDFEmptyError(ERROR_MESSAGES['pdf_empty'])

                    if page_errors:
                        self.stats['warnings'].extend(page_errors)

                    self.text = ' '.join(extracted_text)
                    self.stats['total_characters'] = len(self.text)
                    self.stats['success'] = True
                    
                    if self.text:
                        self.language = self.detect_language(self.text[:1000])
                        self.stats['detected_language'] = self.language
                    
                    logger.info("Text extraction completed successfully")
                    return self.text, self.stats
                    
                except PDFError:
                    raise
                except Exception as e:
                    logger.error(f"Error during text extraction: {str(e)}")
                    raise PDFCorruptedError(f"{ERROR_MESSAGES['pdf_corrupted']} Details: {str(e)}")

        except Exception as e:
            self.stats['error'] = str(e)
            self.stats['success'] = False
            logger.error(f"PDF processing error: {str(e)}")
            raise
            
        finally:
            # Cleanup temporary files
            try:
                import shutil
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    logger.debug("Temporary files cleaned up")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary files: {str(e)}")

    def detect_language(self, text: str) -> str:
        """
        Detect text language with error handling.
        """
        if not text.strip():
            logger.warning("Empty text provided for language detection")
            return DEFAULT_LANGUAGE

        try:
            detected = detect(text)
            if detected in SUPPORTED_LANGUAGES:
                return detected
            
            warning = f"Unsupported language detected: {detected}"
            logger.warning(warning)
            self.stats['warnings'].append(warning)
            return DEFAULT_LANGUAGE

        except LangDetectException as e:
            warning = f"Language detection failed: {str(e)}"
            logger.warning(warning)
            self.stats['warnings'].append(warning)
            return DEFAULT_LANGUAGE

    def clean_text(self, text: str) -> str:
        """
        Clean text with error tracking.
        """
        if not text:
            return ""

        try:
            # Remove multiple whitespace characters
            text = ' '.join(text.split())
            # Remove special characters
            replacements = {
                '\xa0': ' ',  # non-breaking space
                '\f': ' ',    # form feed
                '\t': ' ',    # tab
                '\v': ' ',    # vertical tab
                '\r': ' '     # carriage return
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
                
            return text.strip()

        except Exception as e:
            warning = f"Text cleaning error: {str(e)}"
            logger.warning(warning)
            self.stats['warnings'].append(warning)
            return text.strip()

    def get_text_chunks(self, chunk_size: int = PDF_CHUNK_SIZE) -> list:
        """
        Get text chunks with validation.
        """
        if not self.text:
            return []
            
        try:
            chunks = []
            words = self.text.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    current_chunk.append(word)
                    current_length += word_length
                    
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating text chunks: {str(e)}")
            return [self.text]

    def get_summary(self) -> Dict:
        """
        Get processing summary with error details.
        """
        return {
            'filename': self.pdf_path.name,
            'language': self.language,
            'statistics': self.stats,
            'has_warnings': bool(self.stats['warnings']),
            'warning_count': len(self.stats['warnings'])
        }

if __name__ == "__main__":
    import sys
    
    # Set up logging for main execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        if len(sys.argv) > 1:
            pdf_path = sys.argv[1]
        else:
            pdf_path = "example.pdf"
            
        processor = PDFProcessor(pdf_path)
        text, stats = processor.extract_text()
        
        print("\nPDF Processing Summary:")
        summary = processor.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
            
    except PDFError as e:
        print(f"\nPDF Processing Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error: {str(e)}")
        sys.exit(1)