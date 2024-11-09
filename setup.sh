#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Keyword Extraction System...${NC}"

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
    # Try multiple potential certificate locations
    cert_script_locations=(
        "/Applications/Python*/Install Certificates.command"
        "$HOME/Library/Python/*/Install Certificates.command"
        "./venv/lib/python*/Install Certificates.command"
    )
    
    cert_installed=false
    for location in "${cert_script_locations[@]}"; do
        if ls $location >/dev/null 2>&1; then
            echo -e "${BLUE}Found certificates script at: ${location}${NC}"
            sudo $location || true  # Continue even if it fails
            cert_installed=true
            break
        fi
    done
    
    if [ "$cert_installed" = false ]; then
        echo -e "${RED}Warning: Could not find Install Certificates.command${NC}"
        echo -e "${BLUE}Installing certifi as fallback...${NC}"
        pip install --upgrade certifi
    fi
fi

# Install requirements with error handling
echo -e "${BLUE}Installing requirements...${NC}"
if ! pip install -r requirements.txt; then
    echo -e "${RED}Failed to install requirements. Please check the error message above.${NC}"
    exit 1
fi

# Create NLTK setup script
echo -e "${BLUE}Creating NLTK setup script...${NC}"
cat > setup_nltk.py << 'EOL'
import ssl
import nltk
from pathlib import Path
import sys

def setup_nltk():
    # Handle SSL certificate verification
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Set up NLTK data directory
    nltk_dir = Path.home() / 'nltk_data'
    nltk_dir.mkdir(exist_ok=True)

    # Download required NLTK data
    required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    
    for data in required_data:
        try:
            print(f"Downloading {data}...")
            nltk.download(data, download_dir=str(nltk_dir), quiet=False)
        except Exception as e:
            print(f"Failed to download {data} to home directory: {str(e)}")
            try:
                # Try alternative location
                alt_dir = Path(__file__).parent / 'nltk_data'
                alt_dir.mkdir(exist_ok=True)
                nltk.download(data, download_dir=str(alt_dir), quiet=False)
                print(f"Successfully downloaded {data} to alternate location")
            except Exception as e2:
                print(f"All download attempts failed for {data}: {str(e2)}")
                sys.exit(1)

if __name__ == '__main__':
    setup_nltk()
EOL

# Run NLTK setup
echo -e "${BLUE}Downloading NLTK data...${NC}"
if ! python setup_nltk.py; then
    echo -e "${RED}Failed to download NLTK data. Please check the error messages above.${NC}"
    echo -e "${RED}You may need to run setup_nltk.py manually after fixing any SSL issues.${NC}"
    # Continue with setup despite NLTK download failure
fi

# Create __init__.py files
echo -e "${BLUE}Setting up project files...${NC}"
touch modules/__init__.py
touch config/__init__.py
touch gui/__init__.py
touch tests/__init__.py

# Create gui module structure
echo -e "${BLUE}Setting up GUI module...${NC}"
cat > gui/__init__.py << EOL
"""
GUI module for the keyword extraction system.
Contains all GUI-related components and handlers.
"""
from .app_window import AppWindow

__all__ = ['AppWindow']
EOL

# Create .env file template
echo -e "${BLUE}Creating .env template...${NC}"
cat > .env.template << EOL
# API Configuration
AI_BASE_URL=http://localhost:1234/v1
AI_API_KEY=your-api-key-here
AI_MODEL=your-model-name

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=10485760  # 10MB in bytes
EOL

# Create basic test file
echo -e "${BLUE}Creating basic test setup...${NC}"
cat > tests/test_basic.py << EOL
def test_imports():
    """Test that all main modules can be imported."""
    try:
        from modules.pdf_processor import PDFProcessor
        from modules.text_analyzer import TextAnalyzer
        from modules.ai_handler import AIHandler
        from gui.app_window import AppWindow
        assert True
    except ImportError as e:
        assert False, f"Import failed: {str(e)}"
EOL

# Create .gitignore
echo -e "${BLUE}Creating .gitignore...${NC}"
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Project specific
output/
temp/
cache/
logs/
.env
nltk_data/

# IDE
.idea/
.vscode/
*.swp
*.swo

# macOS
.DS_Store
EOL

# Create directory structure guide
echo -e "${BLUE}Creating project structure documentation...${NC}"
cat > PROJECT_STRUCTURE.md << EOL
# Project Structure

\`\`\`
keyword_extraction/
├── config/           # Configuration files and settings
│   ├── __init__.py
│   └── config.py
├── gui/             # GUI components
│   ├── __init__.py
│   └── app_window.py
├── modules/         # Core processing modules
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── text_analyzer.py
│   └── ai_handler.py
├── tests/           # Test files
│   ├── __init__.py
│   └── test_basic.py
├── output/          # Generated output files
├── temp/            # Temporary files
├── cache/           # Cache files
├── logs/            # Log files
├── main.py          # Application entry point
├── setup.sh         # Setup script
├── requirements.txt # Project dependencies
└── .env            # Environment variables
\`\`\`
EOL

echo -e "${GREEN}Setup complete! Your development environment is ready.${NC}"
echo -e "${BLUE}Additional steps you might want to take:${NC}"
echo -e "1. Copy .env.template to .env and configure your settings"
echo -e "2. Run tests with 'pytest tests/'"
echo -e "3. If NLTK data download failed, try running 'python setup_nltk.py' manually"
echo -e "\n${GREEN}Project structure has been documented in PROJECT_STRUCTURE.md${NC}"
echo -e "\n${GREEN}To activate the virtual environment, run:${NC}"
echo -e "source venv/bin/activate"
echo -e "${GREEN}To deactivate, simply run:${NC}"
echo -e "deactivate"