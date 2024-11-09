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

# Check available disk space (minimum 1GB)
available_space=$(df -Pk . | tail -1 | awk '{print $4}')
if [ $available_space -lt 1048576 ]; then  # 1GB in KB
    echo -e "${RED}Insufficient disk space. At least 1GB required.${NC}"
    exit 1
fi

# Create project directory structure if not exists
echo -e "${BLUE}Creating project structure...${NC}"
for dir in config modules gui output temp cache logs; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}Created directory: $dir${NC}"
    else
        echo -e "${BLUE}Directory already exists: $dir${NC}"
    fi
done

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
langdetect>=1.0.9

# Text processing
nltk==3.8.1  # Spezifische Version für bessere Kompatibilität
matplotlib>=3.0.0
wordcloud>=1.9.0

# AI
openai>=1.3.0
tenacity>=8.2.0

# Utilities
python-dotenv>=1.0.0
EOL

# Install requirements with error handling and progress display
echo -e "${BLUE}Installing requirements...${NC}"
if ! pip install -r requirements.txt; then
    echo -e "${RED}Failed to install requirements. Please check the error message above.${NC}"
    exit 1
fi

# Create simplified NLTK setup script
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

    required_data = ['punkt', 'stopwords']
    
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

# Create simplified .env template with essential settings
echo -e "${BLUE}Creating .env template...${NC}"
cat > .env.template << EOL
# AI Configuration
AI_BASE_URL=http://localhost:1234/v1
AI_API_KEY=your-api-key-here
AI_MODEL=your-model-name
EOL

# Backup existing .env if it exists
if [ -f .env ]; then
    echo -e "${BLUE}Backing up existing .env file...${NC}"
    cp .env .env.backup
fi

# Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env from template...${NC}"
    cp .env.template .env
fi

echo -e "${GREEN}Setup complete! Your development environment is ready.${NC}"
echo -e "${BLUE}Additional steps:${NC}"
echo -e "1. Configure your API settings in .env"
echo -e "2. For development, activate the virtual environment with:"
echo -e "   source venv/bin/activate"