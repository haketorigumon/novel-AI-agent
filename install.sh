#!/bin/bash

# Novel AI Agent Installation Script

echo "🤖 Installing Novel AI Agent..."

# Check if Python 3.8+ is installed
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version detected"
else
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p output
mkdir -p backups
mkdir -p templates

# Copy environment file
if [ ! -f .env ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    
    # Check if llama3 model is available
    if ollama list | grep -q "llama3"; then
        echo "✅ Llama3 model is available"
    else
        echo "📥 Downloading Llama3 model (this may take a while)..."
        ollama pull llama3
    fi
else
    echo "⚠️ Ollama is not installed. Please install it from https://ollama.ai/"
    echo "After installing Ollama, run: ollama pull llama3"
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit the .env file with your configuration"
echo "3. Make sure Ollama is running: ollama serve"
echo "4. Run the agent: python main.py generate --web-interface"
echo ""
echo "Or start the web interface directly:"
echo "python main.py web"
echo ""
echo "Visit http://localhost:12000 to access the dashboard"