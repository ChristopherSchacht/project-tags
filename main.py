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
from PyQt6.QtWidgets import QApplication

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Project imports
from gui.app_window import AppWindow
from config.config import (
    DEFAULT_METADATA,
    OUTPUT_DIR,
    AI_SETTINGS,
    GENERATE_WORD_CLOUD,
    DEFAULT_LANGUAGE
)
from modules.pdf_processor import PDFProcessor, PDFError
from modules.text_analyzer import TextAnalyzer, TextAnalysisError
from modules.ai_handler import AIHandler, AIError
from modules.utils import setup_logging, safe_file_write, get_timestamp

# Initialize logger
logger = setup_logging(__name__)

# main.py modifications

class KeywordExtractor:
    def __init__(self):
        """Initialize with resource management."""
        self.ai_handler = AIHandler()
        self.processing = False
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async def process_document(self, pdf_path: Path, metadata: Dict, message_queue: asyncio.Queue) -> None:
        if self.processing:
            await message_queue.put({'action': 'update_results', 'text': "Processing already in progress\n"})
            return

        self.processing = True
        analyzer = None

        try:
            # Process PDF
            logger.debug("Starting PDF processing...")
            await message_queue.put({'action': 'set_status', 'text': "Processing document..."})
            await message_queue.put({
                'action': 'update_results', 
                'text': "\n█ Reading PDF...\n"
            })

            # PDF Processor handles its own cleanup in extract_text
            pdf_processor = PDFProcessor(pdf_path)
            text, stats = pdf_processor.extract_text()

            if not stats['success']:
                raise PDFError(stats['error'])
            
            # Get detected language from PDF processing
            detected_language = stats.get('detected_language', DEFAULT_LANGUAGE)
            logger.debug(f"Detected language: {detected_language}")

            logger.debug(f"PDF text extracted, length: {len(text)}")

            # Log any warnings from PDF processing
            if stats.get('warnings'):
                for warning in stats['warnings']:
                    logger.warning(f"PDF Processing warning: {warning}")
                    await message_queue.put({
                        'action': 'update_results',
                        'text': f"⚠️  Warning: {warning}\n"
                    })

            # Analyze text
            await message_queue.put({
                'action': 'update_results', 
                'text': "\n█ Analyzing text...\n"
            })
            analyzer = TextAnalyzer(language=detected_language)   # Language will be auto-detected
            analysis_results = analyzer.analyze_text(text, generate_wordcloud=GENERATE_WORD_CLOUD)

            logger.debug("Text analysis complete")

            # Extract keywords using AI
            await message_queue.put({
                'action': 'update_results', 
                'text': "\n█ Extracting keywords...\n"
            })

            # Get max text length from settings
            max_chars = AI_SETTINGS.get('max_text_chars', 100000)
            truncated_text = text[:max_chars]
            if len(text) > max_chars:
                logger.warning(f"Text truncated from {len(text)} to {max_chars} characters")
                await message_queue.put({
                    'action': 'update_results',
                    'text': "⚠️  Notice: Text was truncated due to length limits\n"
                })

            logger.debug("Calling AI handler...")
            keywords_result = await self.ai_handler.extract_keywords(
                metadata,
                {
                    'text': truncated_text,
                    'analysis': analysis_results
                }
            )

            logger.debug("AI processing complete, preparing results...")

            # Process results
            if not keywords_result.get('success'):
                raise AIError(keywords_result.get('error', 'Unknown AI error'))
                
            # Save results
            timestamp = get_timestamp()
            results = {
                'metadata': metadata,
                'statistics': stats,
                'analysis': analysis_results,
                'tags': keywords_result.get('tags', []),
                'processing_time': keywords_result.get('processing_time', 0),
                'detected_language': keywords_result.get('metadata', {}).get('detected_language', 'unknown')
            }
            
            output_path = OUTPUT_DIR / f'results_{timestamp}.json'
            safe_file_write(results, output_path)
            
            # Display results with improved formatting
            await message_queue.put({
                'action': 'update_results', 
                'text': "\n" + "="*50 + "\n✨ Processing complete!\n" + "="*50 + "\n\n"
            })
            
            # Display tags in a visually distinct way
            await message_queue.put({
                'action': 'update_results', 
                'text': "🏷️ EXTRACTED TAGS:\n" + "-"*30 + "\n\n"
            })
            
            # Get tags and calculate spacing
            tags = keywords_result.get('tags', [])
            max_length = max(len(str(tag.get('tag', ''))) for tag in tags if isinstance(tag, dict)) if tags else 0
            
            # Display tags in rows of 7
            tags_per_row = 7
            for i in range(0, len(tags), tags_per_row):
                # Get up to 7 tags for this row
                row_tags = tags[i:i + tags_per_row]
                
                # Format each tag in the row
                row_items = [f"   •  {str(tag.get('tag', '')).strip():<{max_length}}" for tag in row_tags]
                line = "".join(row_items)
                
                await message_queue.put({
                    'action': 'update_results',
                    'text': line + "\n"
                })
                await asyncio.sleep(0.01)

            # Display save location and language
            await message_queue.put({
                'action': 'update_results',
                'text': f"\n{'='*50}\n📁 Results saved to: {output_path}\n"
                       f"🌐 Detected language: {results['detected_language']}\n{'='*50}\n"
            })

            await message_queue.put({'action': 'set_status', 'text': "Processing complete"})

        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            await message_queue.put({
                'action': 'update_results',
                'text': f"❌ Error during processing: {str(e)}\n"
            })
            raise
        finally:
            try:
                # Just clean up matplotlib resources
                import matplotlib.pyplot as plt
                plt.close('all')
                logger.debug("Matplotlib cleanup completed")
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}", exc_info=True)
            finally:
                self.processing = False

def main():
    """Application entry point."""
    try:
        # Set environment variable for macOS
        if sys.platform == 'darwin':
            os.environ['PYTHONUNBUFFERED'] = '1'
        
        # Initialize processor
        processor = KeywordExtractor()
        
        # Create and run PyQt application
        app = QApplication(sys.argv)
        
        # Create main window
        window = AppWindow(
            process_callback=processor.process_document,
            default_metadata=DEFAULT_METADATA
        )
        
        # Show window
        window.run()
        
        # Start application event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise

if __name__ == "__main__":
    main()