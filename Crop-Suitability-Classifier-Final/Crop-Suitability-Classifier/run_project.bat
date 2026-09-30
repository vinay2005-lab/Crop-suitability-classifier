@echo off
echo ==========================================
echo CropSuit - Complete Project Runner
echo ==========================================

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Training model...
python src\train_model.py

echo.
echo Starting Streamlit application...
streamlit run app.py

pause
