"""
Main application module for keyword extraction system with improved thread handling.
"""
import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Optional
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import queue
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Project imports
from config.config import (
    DEFAULT_METADATA,
    OUTPUT_DIR,
    SUPPORTED_LANGUAGES,
    ERROR_MESSAGES,
    SUPPORTED_FILE_TYPES,
    AI_SETTINGS
)
from modules.pdf_processor import PDFProcessor, PDFError
from modules.text_analyzer import TextAnalyzer, TextAnalysisError
from modules.ai_handler import AIHandler, AIError
from modules.utils import setup_logging, safe_file_write, get_timestamp

# Initialize logger
logger = setup_logging(__name__)

class ProcessingError(Exception):
    """Base class for processing errors."""
    pass

class KeywordExtractionApp:
    """Main application class for keyword extraction system."""
    
    def __init__(self):
        """Initialize the application and its components."""
        self.root = None
        self.message_queue = queue.Queue()
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize processing components and state variables."""
        self.pdf_path = None
        self.processing = False
        self.metadata = DEFAULT_METADATA.copy()
        self.ai_handler = AIHandler()
        self.current_language = SUPPORTED_LANGUAGES[0]
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def setup_main_window(self):
        """Set up the main application window."""
        if self.root is None:
            self.root = tk.Tk()
            
        self.root.title("Keyword Extraction System")
        self.root.geometry("900x700")
        
        # Bring window to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Set application style
        self.style = ttk.Style()
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Status.TLabel', font=('Helvetica', 10))

    def create_ui(self):
        """Create the user interface."""
        self.setup_main_window()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.create_file_section()
        self.create_metadata_section()
        self.create_processing_section()
        self.create_results_section()
        self.create_status_bar()
        
    def create_file_section(self):
        """Create file selection section."""
        file_frame = ttk.LabelFrame(self.main_frame, text="Document Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        ttk.Label(file_frame, text="PDF File:").grid(row=0, column=0, sticky="w")
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, sticky="w")
        
        ttk.Button(file_frame, text="Browse", command=self.select_file).grid(row=0, column=2)
        
        # Language selection
        ttk.Label(file_frame, text="Language:").grid(row=1, column=0, sticky="w")
        self.language_var = tk.StringVar(value=self.current_language)
        language_combo = ttk.Combobox(
            file_frame,
            textvariable=self.language_var,
            values=SUPPORTED_LANGUAGES,
            state="readonly"
        )
        language_combo.grid(row=1, column=1, sticky="w")
        
    def create_metadata_section(self):
        """Create metadata input section."""
        metadata_frame = ttk.LabelFrame(self.main_frame, text="Document Metadata", padding="5")
        metadata_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Create and store metadata entry fields
        self.metadata_entries = {}
        row = 0
        
        for field in DEFAULT_METADATA.keys():
            label = field.replace('_', ' ').title()
            ttk.Label(metadata_frame, text=f"{label}:").grid(row=row, column=0, sticky="w")
            
            if isinstance(DEFAULT_METADATA[field], list):
                # Create entry for list fields (comma-separated)
                entry = ttk.Entry(metadata_frame, width=50)
                entry.grid(row=row, column=1, sticky="ew")
                self.metadata_entries[field] = entry
            else:
                # Create entry for single value fields
                entry = ttk.Entry(metadata_frame, width=50)
                entry.grid(row=row, column=1, sticky="ew")
                self.metadata_entries[field] = entry
            
            row += 1
            
    def create_processing_section(self):
        """Create processing controls section."""
        processing_frame = ttk.Frame(self.main_frame)
        processing_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.process_button = ttk.Button(
            processing_frame,
            text="Extract Keywords",
            command=self.start_processing
        )
        self.process_button.pack(pady=5)
        
        self.progress = ttk.Progressbar(
            processing_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=5)
        
    def create_results_section(self):
        """Create results display section."""
        results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=5)
        
        # Configure grid weights for expansion
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = tk.Text(
            results_frame,
            height=15,
            width=70,
            wrap=tk.WORD
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            results_frame,
            orient="vertical",
            command=self.results_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text['yscrollcommand'] = scrollbar.set
        
    def create_status_bar(self):
        """Create status bar."""
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            style='Status.TLabel'
        )
        self.status_bar.grid(row=1, column=0, sticky="ew")
        
    def select_file(self):
        """Handle file selection."""
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf")]
        )
        if file_path:
            self.pdf_path = Path(file_path)
            self.file_label.config(text=self.pdf_path.name)
            
    def get_metadata(self) -> Dict:
        """Collect metadata from input fields."""
        metadata = {}
        for field, entry in self.metadata_entries.items():
            value = entry.get().strip()
            if isinstance(DEFAULT_METADATA[field], list):
                metadata[field] = [v.strip() for v in value.split(',') if v.strip()]
            else:
                metadata[field] = value
        return metadata

    def update_ui(self):
        """Process any pending UI updates from the message queue."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                action = message.get('action')
                
                if action == 'set_status':
                    self.status_var.set(message['text'])
                elif action == 'update_results':
                    self.results_text.delete('1.0', tk.END)
                    self.results_text.insert(tk.END, message['text'])
                elif action == 'enable_button':
                    self.process_button.state(['!disabled'])
                elif action == 'disable_button':
                    self.process_button.state(['disabled'])
                elif action == 'stop_progress':
                    self.progress.stop()
                elif action == 'show_error':
                    messagebox.showerror("Error", message['text'])
                    
        except queue.Empty:
            pass
        finally:
            # Schedule the next update
            if self.root:
                self.root.after(100, self.update_ui)
        
    async def process_document(self):
        """Process the document and extract keywords."""
        try:
            # Update UI state through queue
            self.message_queue.put({'action': 'set_status', 'text': "Processing document..."})
            self.message_queue.put({'action': 'update_results', 'text': "Reading PDF...\n"})
            
            # Process PDF
            pdf_processor = PDFProcessor(self.pdf_path)
            text, stats = pdf_processor.extract_text()
            
            if not stats['success']:
                raise ProcessingError(stats['error'])
            
            # Analyze text
            self.message_queue.put({'action': 'update_results', 'text': "Analyzing text...\n"})
            analyzer = TextAnalyzer(language=self.language_var.get())
            analysis_results = analyzer.analyze_text(text)
            
            # Generate word cloud
            self.message_queue.put({'action': 'update_results', 'text': "Generating word cloud...\n"})
            wordcloud_path = analyzer.generate_wordcloud(
                analysis_results['word_frequencies']
            )
            
            # Extract keywords
            self.message_queue.put({'action': 'update_results', 'text': "Extracting keywords...\n"})
            metadata = self.get_metadata()
            keywords_result = await self.ai_handler.extract_keywords(
                metadata,
                {
                    'text': text[:AI_SETTINGS['max_text_chars']],
                    'analysis': analysis_results
                }
            )
            
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
            
            # Display results through queue
            result_text = "Processing complete!\n\nExtracted Keywords:\n"
            for kw in keywords_result['keywords']:
                result_text += f"• {kw['keyword']} (Relevance: {kw['relevance']:.2f})\n"
            result_text += f"\nResults saved to: {output_path}\n"
            result_text += f"Word cloud saved to: {wordcloud_path}\n"
            
            self.message_queue.put({'action': 'update_results', 'text': result_text})
            self.message_queue.put({'action': 'set_status', 'text': "Processing complete"})
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            self.message_queue.put({'action': 'set_status', 'text': "Error occurred"})
            self.message_queue.put({'action': 'show_error', 'text': str(e)})

    def start_processing(self):
        """Start document processing in a separate thread."""
        if not self.pdf_path or self.processing:
            return
            
        if not self.validate_inputs():
            return
            
        self.processing = True
        self.message_queue.put({'action': 'disable_button'})
        self.progress.start()
        
        # Create and start processing thread
        async def process_wrapper():
            try:
                await self.process_document()
            finally:
                self.processing = False
                self.message_queue.put({'action': 'enable_button'})
                self.message_queue.put({'action': 'stop_progress'})
                
        threading.Thread(
            target=lambda: asyncio.run(process_wrapper()),
            daemon=True  # Make thread daemon so it doesn't prevent app exit
        ).start()
        
    def validate_inputs(self) -> bool:
        """Validate user inputs."""
        if not self.pdf_path:
            messagebox.showerror("Error", "Please select a PDF file")
            return False
            
        if not self.pdf_path.exists():
            messagebox.showerror("Error", "Selected file no longer exists")
            return False
            
        return True
        
    def run(self):
        """Run the application."""
        try:
            self.create_ui()
            
            # Start UI update loop
            self.update_ui()
            
            # Start main event loop
            self.root.mainloop()
                
        except Exception as e:
            logger.critical(f"Application crash: {str(e)}")
            messagebox.showerror("Critical Error", f"Application crashed: {str(e)}")
            raise

def main():
    """Application entry point."""
    try:
        # Set environment variable for macOS
        if sys.platform == 'darwin':
            os.environ['PYTHONUNBUFFERED'] = '1'
            
        app = KeywordExtractionApp()
        app.run()
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise

if __name__ == "__main__":
    main()