# gui/app_window.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import asyncio
import queue
import threading
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class AppWindow:
    """Main application window handler."""
    def __init__(self, process_callback: Callable, supported_languages: list, default_metadata: dict):
        """
        Initialize the application window.
        
        Args:
            process_callback: Callback function for document processing
            supported_languages: List of supported languages
            default_metadata: Default metadata dictionary
        """
        self.process_callback = process_callback
        self.supported_languages = supported_languages
        self.default_metadata = default_metadata
        
        self.pdf_path: Optional[Path] = None
        self.processing = False
        self.message_queue = queue.Queue()
        
        # Initialize main window
        self.root = tk.Tk()
        self.setup_window()
        self.create_ui()
        
        # Start UI update loop
        self.schedule_ui_updates()
        
    def setup_window(self):
        """Configure the main window properties."""
        self.root.title("Keyword Extraction System")
        self.root.geometry("900x700")
        
        # Configure window appearance
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Set window style
        self.style = ttk.Style()
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Status.TLabel', font=('Helvetica', 10))
        
        # Ensure window appears on top when launched
        self.root.focus_force()
        
    def create_ui(self):
        """Create all UI elements."""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        self._create_file_section()
        self._create_metadata_section()
        self._create_processing_section()
        self._create_results_section()
        self._create_status_bar()
        
    def _create_file_section(self):
        """Create file selection section."""
        file_frame = ttk.LabelFrame(self.main_frame, text="Document Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        
        # File selection
        ttk.Label(file_frame, text="PDF File:").grid(row=0, column=0, sticky="w")
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Button(
            file_frame,
            text="Browse",
            command=self._select_file
        ).grid(row=0, column=2, padx=5)
        
        # Language selection
        ttk.Label(file_frame, text="Language:").grid(row=1, column=0, sticky="w")
        self.language_var = tk.StringVar(value=self.supported_languages[0])
        language_combo = ttk.Combobox(
            file_frame,
            textvariable=self.language_var,
            values=self.supported_languages,
            state="readonly"
        )
        language_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
    def _create_metadata_section(self):
        """Create metadata input section."""
        metadata_frame = ttk.LabelFrame(self.main_frame, text="Document Metadata", padding="5")
        metadata_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.metadata_entries = {}
        for idx, (field, default_value) in enumerate(self.default_metadata.items()):
            label = field.replace('_', ' ').title()
            ttk.Label(metadata_frame, text=f"{label}:").grid(
                row=idx, column=0, sticky="w", padx=5, pady=2
            )
            
            entry = ttk.Entry(metadata_frame, width=50)
            entry.grid(row=idx, column=1, sticky="ew", padx=5, pady=2)
            self.metadata_entries[field] = entry
            
            if isinstance(default_value, list):
                entry.insert(0, ', '.join(default_value))
            else:
                entry.insert(0, str(default_value))
                
    def _create_processing_section(self):
        """Create processing controls section."""
        processing_frame = ttk.Frame(self.main_frame)
        processing_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.process_button = ttk.Button(
            processing_frame,
            text="Extract Keywords",
            command=self._start_processing
        )
        self.process_button.pack(pady=5)
        
        self.progress = ttk.Progressbar(
            processing_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=5)
        
    def _create_results_section(self):
        """Create results display section."""
        results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=5)
        
        # Make results section expandable
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create text widget with scrollbar
        self.results_text = tk.Text(
            results_frame,
            height=15,
            width=70,
            wrap=tk.WORD
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(
            results_frame,
            orient="vertical",
            command=self.results_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text['yscrollcommand'] = scrollbar.set
        
    def _create_status_bar(self):
        """Create status bar."""
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            style='Status.TLabel'
        )
        self.status_bar.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        
    def _select_file(self):
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
            if isinstance(self.default_metadata[field], list):
                metadata[field] = [v.strip() for v in value.split(',') if v.strip()]
            else:
                metadata[field] = value
        return metadata
        
    def _validate_inputs(self) -> bool:
        """Validate user inputs."""
        if not self.pdf_path:
            messagebox.showerror("Error", "Please select a PDF file")
            return False
            
        if not self.pdf_path.exists():
            messagebox.showerror("Error", "Selected file no longer exists")
            return False
            
        return True
        
    def _start_processing(self):
        """Start document processing."""
        if not self.pdf_path or self.processing:
            return
            
        if not self._validate_inputs():
            return
            
        self.processing = True
        self.message_queue.put({'action': 'disable_button'})
        self.progress.start()
        
        # Create processing thread using asyncio
        async def process_wrapper():
            try:
                await self.process_callback(
                    self.pdf_path,
                    self.get_metadata(),
                    self.language_var.get(),
                    self.message_queue
                )
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                self.message_queue.put({
                    'action': 'show_error',
                    'text': f"Processing error: {str(e)}"
                })
            finally:
                self.processing = False
                self.message_queue.put({'action': 'enable_button'})
                self.message_queue.put({'action': 'stop_progress'})
                
        threading.Thread(
            target=lambda: asyncio.run(process_wrapper()),
            daemon=True
        ).start()
        
    def schedule_ui_updates(self):
        """Schedule periodic UI updates."""
        def update():
            try:
                while True:
                    message = self.message_queue.get_nowait()
                    self._handle_ui_message(message)
            except queue.Empty:
                pass
            finally:
                # Schedule next update using a shorter interval for better responsiveness
                if self.root:
                    self.root.after(50, update)
                    
        # Start the update loop
        self.root.after(50, update)
        
    def _handle_ui_message(self, message: Dict):
        """Handle UI update messages."""
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
            
    def run(self):
        """Start the main event loop."""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.critical(f"Application crash: {str(e)}")
            messagebox.showerror("Critical Error", f"Application crashed: {str(e)}")
            raise