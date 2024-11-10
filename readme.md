# Keyword Extraction System

This project is a Python-based system for creating descriptive tags for PDF documents via a mix of text-statistics and ai.

<img width="903" alt="image" src="https://github.com/user-attachments/assets/8f83743c-4949-4c00-a87a-f7bd26653243">


## Prerequisites

- Python 3.8 or higher
- macOS or Linux

## Quick Setup

1. Clone the repository and navigate to the project directory
2. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```
3. Run the setup script:
   ```bash
   ./setup.sh
   ```
4. Configure your API settings:
   ```bash
   # Edit .env with your API settings
   AI_BASE_URL=your-base-url
   AI_API_KEY=your-api-key
   AI_MODEL=your-model
   ```

## Project Structure

```
project-directory/
├── main.py
├── setup.sh
├── .env
├── config/
│   ├── __init__.py
│   ├── config.py
│   ├── stopwords_de.txt
│   └── stopwords_en.txt
├── modules/
│   ├── __init__.py
│   ├── ai_handler.py
│   ├── pdf_processor.py
│   ├── text_analyzer.py
│   └── utils.py
├── output/
├── temp/
├── cache/
└── logs/
```

## Running the Application

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Start the application:
   ```bash
   python main.py
   ```

## Features

- PDF text extraction with support for multiple languages
- Advanced text analysis and keyword extraction
- Modern GUI interface
- AI-powered tag generation
- Automatic language detection
- Support for both English and German documents

## Troubleshooting

If you encounter any issues:

1. Check the logs in the `logs/` directory
2. Ensure your Python version is 3.8 or higher:
   ```bash
   python3 --version
   ```
3. Verify the virtual environment is activated:
   ```bash
   which python
   # Should show path to your venv/bin/python
   ```
4. Make sure your `.env` file contains valid API credentials

## Common Issues

1. **Module not found errors**: Ensure you're running Python from the virtual environment
2. **Permission denied**: Run `chmod +x setup.sh` to make the setup script executable
3. **NLTK data download fails**: Check your internet connection
4. **API errors**: Verify your API credentials in the `.env` file
