# gui/app_window.py

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit,
    QProgressBar, QTextEdit, QScrollArea, QFrame, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
import asyncio
from pathlib import Path
from typing import Dict, Optional
import logging
import queue

logger = logging.getLogger(__name__)

class AsyncWorker(QThread):
    """Asynchronous worker thread for document processing."""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(dict)
    
    def __init__(self, process_func, pdf_path, metadata):  # Removed language parameter
        super().__init__()
        self.process_func = process_func
        self.pdf_path = pdf_path
        self.metadata = metadata
        self.loop = None
        
    def run(self):
        """Execute the processing function in a separate thread."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Create a queue for progress updates
            progress_queue = asyncio.Queue()
            
            # Run the processing function and handle messages
            async def process_and_handle_messages():
                try:
                    # Process messages from the queue continuously
                    async def message_handler():
                        while True:
                            try:
                                message = await progress_queue.get()
                                self.progress.emit(message)
                            except Exception as e:
                                logger.error(f"Error handling message: {str(e)}")
                                break

                    # Start message handler task
                    message_handler_task = asyncio.create_task(message_handler())
                    
                    # Run the main processing function (removed language parameter)
                    await self.process_func(
                        self.pdf_path,
                        self.metadata,
                        progress_queue
                    )
                    
                    # Allow time for final messages to be processed
                    await asyncio.sleep(0.1)
                    
                    # Cancel message handler
                    message_handler_task.cancel()
                    try:
                        await message_handler_task
                    except asyncio.CancelledError:
                        pass
                        
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    self.error.emit(str(e))

            self.loop.run_until_complete(process_and_handle_messages())
            self.finished.emit()
            
        except Exception as e:
            logger.error(f"Worker thread error: {str(e)}")
            self.error.emit(str(e))
        finally:
            if self.loop:
                self.loop.close()

class AppWindow(QMainWindow):
    """Modern macOS-styled main window."""
    
    def __init__(self, process_callback, default_metadata):  # Removed supported_languages parameter
        super().__init__()
        self.process_callback = process_callback
        self.default_metadata = default_metadata
        self.pdf_path = None
        self.worker = None
        self.message_queue = queue.Queue()
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface with modern macOS styling."""
        self.setWindowTitle("Document Analyzer")
        self.setMinimumSize(900, 700)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Set modern styling
        self.setup_styling()
        
        # Create UI sections
        self.create_file_section(layout)
        self.create_metadata_section(layout)
        self.create_processing_section(layout)
        self.create_results_section(layout)
        
        # Initialize status bar
        self.statusBar().showMessage("Ready")
        
        # Setup periodic UI updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(50)  # 50ms interval
        
    def setup_styling(self):
        """Apply modern macOS-like styling."""
        # Set font
        font = QFont(".AppleSystemUIFont", 13)
        QApplication.setFont(font)
        
        # Set color scheme
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.Base, QColor(249, 249, 249))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        self.setPalette(palette)
        
        # Set stylesheet for modern look
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
            }
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0069D9;
            }
            QPushButton:pressed {
                background-color: #0062CC;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #F0F0F0;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007AFF;
                border-radius: 4px;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                background-color: white;
            }
            QTextEdit {
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                background-color: white;
                padding: 8px;
            }
        """)
        
    def create_file_section(self, parent_layout):
        """Create file selection section."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        browse_button = QPushButton("Choose PDF")
        browse_button.clicked.connect(self.select_file)
        file_layout.addWidget(QLabel("PDF File:"))
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(browse_button)
        
        layout.addLayout(file_layout)
        parent_layout.addWidget(frame)
        
    def create_metadata_section(self, parent_layout):
        """Create metadata input section."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)
        
        self.metadata_entries = {}
        for field, default_value in self.default_metadata.items():
            field_layout = QHBoxLayout()
            label = QLabel(field.replace('_', ' ').title())
            entry = QLineEdit()
            
            if isinstance(default_value, list):
                entry.setText(', '.join(default_value))
            else:
                entry.setText(str(default_value))
                
            self.metadata_entries[field] = entry
            field_layout.addWidget(label)
            field_layout.addWidget(entry)
            layout.addLayout(field_layout)
            
        parent_layout.addWidget(frame)
        
    def create_processing_section(self, parent_layout):
        """Create processing controls section."""
        layout = QHBoxLayout()
        
        self.process_button = QPushButton("Process Document")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setFixedHeight(40)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        
        layout.addWidget(self.process_button)
        layout.addWidget(self.progress_bar)
        parent_layout.addLayout(layout)
        
    def create_results_section(self, parent_layout):
        """Create results display section with QPlainTextEdit."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)

        self.results_text = QPlainTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumBlockCount(10000)
        self.results_text.setMaximumHeight(2000)
        self.results_text.setUndoRedoEnabled(False)

        scroll.setWidget(self.results_text)
        parent_layout.addWidget(scroll, 1)
        
    def select_file(self):
        """Handle file selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF Document",
            str(Path.home()),
            "PDF Files (*.pdf)"
        )
        
        if file_path:
            self.pdf_path = Path(file_path)
            self.file_label.setText(self.pdf_path.name)
            
    def get_metadata(self) -> Dict:
        """Collect metadata from input fields."""
        metadata = {}
        for field, entry in self.metadata_entries.items():
            value = entry.text().strip()
            if isinstance(self.default_metadata[field], list):
                metadata[field] = [v.strip() for v in value.split(',') if v.strip()]
            else:
                metadata[field] = value
        return metadata
        
    def start_processing(self):
        """Start document processing."""
        if not self.pdf_path or self.worker is not None:
            return
            
        self.process_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Create and start worker thread
        self.worker = AsyncWorker(
            self.process_callback,
            self.pdf_path,
            self.get_metadata()
        )
        
        self.worker.finished.connect(self.processing_finished)
        self.worker.error.connect(self.processing_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.start()
        
    def processing_finished(self):
        """Handle processing completion."""
        try:
            self.statusBar().showMessage("Processing complete")
            QApplication.processEvents()
            self.cleanup_worker()
            logger.debug("Processing finished successfully")
            
        except Exception as e:
            logger.error(f"Error in processing_finished: {str(e)}")
            self.processing_error(str(e))
        
    def processing_error(self, error_msg):
        """Handle processing errors."""
        try:
            logger.debug(f"Processing error occurred: {error_msg}")
            self.cleanup_worker()
            self.statusBar().showMessage("Error occurred")
            
            if hasattr(self, 'results_text'):
                self.results_text.appendPlainText(f"Error: {error_msg}\n")
                cursor = self.results_text.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self.results_text.setTextCursor(cursor)
                
        except Exception as e:
            logger.error(f"Error in processing_error handler: {str(e)}", exc_info=True)
            
    def update_progress(self, progress_data):
        """Update UI with progress information."""
        try:
            if not isinstance(progress_data, dict):
                logger.error(f"Invalid progress_data type: {type(progress_data)}")
                return
                
            action = progress_data.get('action')
            text = progress_data.get('text', '')
            
            if not action:
                logger.error("No action specified in progress_data")
                return
                
            if action == 'update_results':
                if hasattr(self, 'results_text'):
                    try:
                        self.results_text.appendPlainText(str(text))
                        cursor = self.results_text.textCursor()
                        cursor.movePosition(cursor.MoveOperation.End)
                        self.results_text.setTextCursor(cursor)
                        QApplication.processEvents()
                    except Exception as e:
                        logger.error(f"Error updating text widget: {str(e)}", exc_info=True)
                        
            elif action == 'set_status':
                try:
                    self.statusBar().showMessage(str(text))
                except Exception as e:
                    logger.error(f"Error updating status bar: {str(e)}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Critical error in update_progress: {str(e)}", exc_info=True)
            
    def cleanup_worker(self):
        """Clean up worker thread and resources."""
        if self.worker:
            try:
                self.worker.finished.disconnect()
                self.worker.error.disconnect()
                self.worker.progress.disconnect()
                
                if self.worker.loop and self.worker.loop.is_running():
                    self.worker.loop.stop()
                
                self.worker.quit()
                if not self.worker.wait(3000):  # 3 second timeout
                    logger.warning("Worker thread did not finish, forcing termination")
                    self.worker.terminate()
                    self.worker.wait()
                    
            except Exception as e:
                logger.error(f"Error during worker cleanup: {str(e)}")
            finally:
                self.worker = None
                self.process_button.setEnabled(True)
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(100)
        
    def update_ui(self):
        """Periodic UI updates."""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self._handle_ui_message(message)
        except queue.Empty:
            pass
            
    def _handle_ui_message(self, message: Dict):
        """Handle UI update messages."""
        try:
            action = message.get('action')
            
            if action == 'set_status':
                self.statusBar().showMessage(message['text'])
            elif action == 'update_results':
                # Ensure text widget exists and is accessible
                if hasattr(self, 'results_text'):
                    self.results_text.append(message['text'])
                    # Force processing of UI events
                    QApplication.processEvents()
            elif action == 'show_error':
                if hasattr(self, 'results_text'):
                    self.results_text.append(f"Error: {message['text']}")
                    QApplication.processEvents()
        except Exception as e:
            logger.error(f"Error handling UI message: {str(e)}")
            
    def run(self):
        """Show the window."""
        self.show()

# Only needed if running this file directly
if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    window = AppWindow(lambda: None, {'field1': 'default'})  # Updated parameters
    window.run()
    sys.exit(app.exec())