@echo off
echo ==========================================
echo      Nexus AI - First Time Setup
echo ==========================================

echo [1/3] Installing Python Dependencies...
pip install -r Requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies!
    pause
    exit /b
)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"


echo [2/3] Checking for Ollama...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama is not installed or not in PATH.
    echo Offline mode will not work nicely without it.
    echo Please install from https://ollama.com/
) else (
    echo Ollama detected! Pulling Llama3 model...
    ollama pull llama3
)

echo [3/3] Creating Directories...
if not exist "Data\Documents" mkdir "Data\Documents"
if not exist "Data\VectorDB" mkdir "Data\VectorDB"

echo ==========================================
echo        Setup Complete!
echo    Run 'python Main.py' to start.
echo ==========================================
pause
