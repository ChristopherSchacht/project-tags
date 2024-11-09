#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Document Analysis System...${NC}"

# Function to check Python version
check_python_version() {
    required_major=3
    required_minor=8
    
    if command -v python3 &> /dev/null; then
        version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        
        if [ "$major" -ge $required_major ] && [ "$minor" -ge $required_minor ]; then
            echo -e "${GREEN}Python $version found${NC}"
            return 0
        fi
    fi
    
    echo -e "${RED}Python $required_major.$required_minor or higher is required${NC}"
    return 1
}

# Check Python version
if ! check_python_version; then
    echo -e "${RED}Please install Python 3.8 or higher first.${NC}"
    exit 1
fi

# Check for Xcode Command Line Tools on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}Checking for Xcode Command Line Tools...${NC}"
    if ! xcode-select -p &>/dev/null; then
        echo -e "${RED}Xcode Command Line Tools not found. Installing...${NC}"
        xcode-select --install
        echo -e "${RED}Please run this script again after Xcode Command Line Tools installation completes.${NC}"
        exit 1
    fi
fi

# Create project directory structure
echo -e "${BLUE}Creating project structure...${NC}"
mkdir -p {config,modules,gui,output,temp,cache,logs,tests}

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install certificates for macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}Installing certificates for macOS...${NC}"
    pip install --upgrade certifi
fi

# Create requirements.txt with updated dependencies
echo -e "${BLUE}Creating requirements.txt...${NC}"
cat > requirements.txt << EOL
# Core dependencies
PyQt6>=6.4.0
PyQt6-Qt6>=6.4.0
PyQt6-sip>=13.4.0

# PDF processing
PyPDF2>=3.0.0
pdfminer.six>=20221105

# Text processing
nltk>=3.8
spacy>=3.5.0
textblob>=0.17.1

# AI and ML
openai>=1.3.0
tenacity>=8.2.0
numpy>=1.24.0
scikit-learn>=1.2.0

# Utilities
python-dotenv>=1.0.0
rich>=13.0.0
loguru>=0.7.0
tqdm>=4.65.0

# Testing
pytest>=7.3.1
pytest-qt>=4.2.0
pytest-asyncio>=0.21.0
EOL

# Install requirements with error handling
echo -e "${BLUE}Installing requirements...${NC}"
if ! pip install -r requirements.txt; then
    echo -e "${RED}Failed to install requirements. Please check the error message above.${NC}"
    exit 1
fi

# Download spaCy model
echo -e "${BLUE}Downloading spaCy language model...${NC}"
python -m spacy download en_core_web_sm

# Create NLTK setup script with updated error handling
echo -e "${BLUE}Setting up NLTK...${NC}"
cat > setup_nltk.py << 'EOL'
import ssl
import nltk
from pathlib import Path
import sys

def setup_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk_dir = Path.home() / 'nltk_data'
    nltk_dir.mkdir(exist_ok=True)

    required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for data in required_data:
        try:
            print(f"Downloading {data}...")
            nltk.download(data, quiet=False)
        except Exception as e:
            print(f"Failed to download {data}: {str(e)}")
            sys.exit(1)

if __name__ == '__main__':
    setup_nltk()
EOL

# Run NLTK setup
echo -e "${BLUE}Downloading NLTK data...${NC}"
if ! python setup_nltk.py; then
    echo -e "${RED}Failed to download NLTK data. Please check the error messages above.${NC}"
    exit 1
fi

# Create test setup for PyQt
echo -e "${BLUE}Creating PyQt test setup...${NC}"
cat > tests/test_gui.py << EOL
import pytest
from PyQt6.QtWidgets import QApplication
from gui.app_window import ModernMacOSWindow

@pytest.fixture
def app(qtbot):
    test_app = QApplication([])
    return test_app

@pytest.fixture
def window(app, qtbot):
    window = ModernMacOSWindow(
        process_callback=lambda x: None,
        supported_languages=['en'],
        default_metadata={}
    )
    window.show()
    qtbot.addWidget(window)
    return window

def test_window_creation(window):
    """Test that the window is created successfully."""
    assert window.isVisible()
    assert window.windowTitle() == "Document Analyzer"
EOL

# Create .env template with updated settings
echo -e "${BLUE}Creating .env template...${NC}"
cat > .env.template << EOL
# Application Settings
DEBUG=False
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=10485760  # 10MB in bytes

# UI Settings
ENABLE_DARK_MODE=True
ENABLE_ANIMATIONS=True
USE_SYSTEM_ACCENT_COLOR=True

# AI Configuration
AI_BASE_URL=http://localhost:1234/v1
AI_API_KEY=your-api-key-here
AI_MODEL=your-model-name
EOL

echo -e "${GREEN}Setup complete! Your development environment is ready.${NC}"
echo -e "${BLUE}Additional steps:${NC}"
echo -e "1. Copy .env.template to .env and configure your settings"
echo -e "2. Run tests with 'pytest tests/'"
echo -e "3. For development, activate the virtual environment with:"
echo -e "   source venv/bin/activate"