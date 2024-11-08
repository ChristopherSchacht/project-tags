# Keyword Extraction System

This project is a Python-based system for extracting keywords from PDF documents, with support for both English and German languages.

## Prerequisites

- Python 3.8 or higher
- macOS 15 or higher
- M2 Max processor (Apple Silicon) compatible

## Setup Instructions

1. Navigate to your project directory (where `main.py` is located).

2. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```

3. Run the setup script:
   ```bash
   ./setup.sh
   ```

The setup script will:
- Create necessary directory structure in your current directory
- Set up a Python virtual environment (venv)
- Install all required dependencies
- Download necessary NLTK data
- Configure the project environment
- Set up development tools and testing infrastructure

## Project Structure

```
project-directory/
├── main.py
├── requirements.txt
├── setup.sh
├── .env.template
├── .env
├── .gitignore
├── venv/
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
├── tests/
│   ├── __init__.py
│   └── test_basic.py
├── output/
├── temp/
├── cache/
└── logs/
```

## Configuration

1. After running the setup script, copy the environment template:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` with your specific settings:
   ```env
   AI_BASE_URL=http://localhost:1234/v1
   AI_API_KEY=your-api-key-here
   AI_MODEL=your-model-name
   DEBUG=False
   LOG_LEVEL=INFO
   MAX_UPLOAD_SIZE=10485760
   ```

## Development Tools

The project includes several development tools:

- **pytest**: For running unit tests
  ```bash
  pytest tests/
  ```

- **black**: For code formatting
  ```bash
  black .
  ```

- **mypy**: For type checking
  ```bash
  mypy .
  ```

## Using the Virtual Environment

To activate the virtual environment:
```bash
source venv/bin/activate
```

To deactivate the virtual environment:
```bash
deactivate
```

## Running the Application

After activating the virtual environment:
```bash
python main.py
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Development Guidelines

1. Always activate the virtual environment before working on the project
2. Format your code using black before committing changes
3. Run type checking with mypy to catch potential type-related issues
4. Add tests for new functionality
5. Keep the `.env` file private and never commit it to version control

## Troubleshooting

If you encounter any issues:

1. Check the logs in the `logs/` directory
2. Ensure your Python version is 3.8 or higher:
   ```bash
   python3 --version
   ```
3. Verify that all dependencies are installed correctly:
   ```bash
   pip list
   ```
4. Make sure your `.env` file is properly configured
5. If you get permission errors, check the file permissions:
   ```bash
   chmod +x setup.sh
   ```
6. If you get import errors, make sure you're running Python from the virtual environment:
   ```bash
   which python
   # Should show path to your venv/bin/python
   ```

## Common Issues

1. **requirements.txt not found**: Make sure you've created the requirements.txt file in your project directory before running setup.sh

2. **Module not found errors**: Ensure you're running Python from the virtual environment and all requirements are installed

3. **Permission denied**: Run `chmod +x setup.sh` to make the setup script executable

4. **NLTK data download fails**: Check your internet connection and try downloading manually:
   ```python
   import nltk
   nltk.download()
   ```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Run tests and ensure they pass
5. Format your code using black
6. Submit a pull request