@echo off
echo ========================================
echo    API Diabetes - Version Simple
echo ========================================

python --version > nul 2>&1
if errorlevel 1 (
    echo Error: Python no encontrado
    pause
    exit /b 1
)

echo Instalando dependencias...
pip install -r requirements.txt

echo.
echo Iniciando API...
echo URL: http://localhost:5000
echo.
python ml_diabetes_simple.py

pause