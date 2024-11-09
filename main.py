# main.py
"""
Main application module for keyword extraction system.
Handles application initialization and processing logic.
"""
import os
import sys
import asyncio
import json
from pathlib import Path
import logging
from typing import Dict
import queue

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Project imports
from gui.app_window import AppWindow
from config.config import (
    DEFAULT_METADATA,
    OUTPUT_DIR,
    SUPPORTED_LANGUAGES,
    AI_SETTINGS
)
from modules.pdf_processor import PDFProcessor, PDFError
from modules.text_analyzer import TextAnalyzer, TextAnalysisError
from modules.ai_handler import AIHandler, AIError
from modules.utils import setup_logging, safe_file_write, get_timestamp

# Initialize logger
logger = setup_logging(__name__)

class KeywordExtractor:
    """Main processor class for keyword extraction."""
    
    def __init__(self):
        """Initialize the keyword extraction processor."""
        self.ai_handler = AIHandler()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async def process_document(
        self,
        pdf_path: Path,
        metadata: Dict,
        language: str,
        message_queue: queue.Queue
    ) -> None:
        """
        Process a document and extract keywords.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Document metadata dictionary
            language: Document language
            message_queue: Queue for UI updates
        """
        try:
            # Process PDF
            message_queue.put({'action': 'set_status', 'text': "Processing document..."})
            message_queue.put({'action': 'update_results', 'text': "Reading PDF...\n"})
            
            pdf_processor = PDFProcessor(pdf_path)
            text, stats = pdf_processor.extract_text()
            
            if not stats['success']:
                raise PDFError(stats['error'])
            
            # Analyze text
            message_queue.put({'action': 'update_results', 'text': "Analyzing text...\n"})
            analyzer = TextAnalyzer(language=language)
            analysis_results = analyzer.analyze_text(text)
            
            # Generate word cloud
            message_queue.put({'action': 'update_results', 'text': "Generating word cloud...\n"})
            wordcloud_path = analyzer.generate_wordcloud(
                analysis_results['word_frequencies']
            )
            
            # Extract keywords using AI
            message_queue.put({'action': 'update_results', 'text': "Extracting keywords...\n"})
            keywords_result = await self.ai_handler.extract_keywords(
                metadata,
                {
                    'text': text[:AI_SETTINGS['max_text_chars']],
                    'analysis': analysis_results
                }
            )
            
            if not keywords_result['success']:
                raise AIError(keywords_result.get('error', 'Unknown AI error'))
            
            # Save results
            timestamp = get_timestamp()
            results = {
                'metadata': metadata,
                'statistics': stats,
                'analysis': analysis_results,
                'keywords': keywords_result['keywords'],
                'wordcloud_path': str(wordcloud_path),
                'processing_time': keywords_result['processing_time']
            }
            
            output_path = OUTPUT_DIR / f'results_{timestamp}.json'
            safe_file_write(results, output_path)
            
            # Display results
            result_text = "Processing complete!\n\nExtracted Keywords:\n"
            for kw in keywords_result['keywords']:
                result_text += f"• {kw['keyword']}\n"
            result_text += f"\nResults saved to: {output_path}\n"
            result_text += f"Word cloud saved to: {wordcloud_path}\n"
            
            # Ensure we send all final updates to the GUI
            message_queue.put({'action': 'update_results', 'text': result_text})
            message_queue.put({'action': 'set_status', 'text': "Processing complete"})
            message_queue.put({'action': 'enable_button'})
            message_queue.put({'action': 'stop_progress'})
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            message_queue.put({
                'action': 'update_results', 
                'text': f"Error during processing: {str(e)}\n"
            })
            message_queue.put({'action': 'set_status', 'text': "Error"})
            message_queue.put({'action': 'enable_button'})
            message_queue.put({'action': 'stop_progress'})
            raise

def main():
    """Application entry point."""
    try:
        # Set environment variable for macOS
        if sys.platform == 'darwin':
            os.environ['PYTHONUNBUFFERED'] = '1'
        
        # Initialize processor
        processor = KeywordExtractor()
        
        # Create and run GUI
        app = AppWindow(
            process_callback=processor.process_document,
            supported_languages=SUPPORTED_LANGUAGES,
            default_metadata=DEFAULT_METADATA
        )
        app.run()
        
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise

if __name__ == "__main__":
    main()