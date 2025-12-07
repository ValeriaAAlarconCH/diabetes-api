# run.ps1
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   API Diabetes - PowerShell Version" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Verificar si Python está instalado
try {
    $pythonVersion = python --version
    Write-Host "✅ Python encontrado: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python no encontrado" -ForegroundColor Red
    Write-Host "💡 Instala Python desde: https://www.python.org/downloads/" -ForegroundColor Yellow
    pause
    exit
}

# Verificar si el entorno virtual existe
if (Test-Path "venv") {
    Write-Host "✅ Entorno virtual encontrado" -ForegroundColor Green
} else {
    Write-Host "⚠️  Creando entorno virtual..." -ForegroundColor Yellow
    python -m venv venv
}

# Activar entorno virtual
Write-Host "`n🔧 Activando entorno virtual..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# Instalar dependencias
Write-Host "📦 Instalando dependencias..." -ForegroundColor Cyan
pip install -r requirements_final.txt

Write-Host "`n🚀 Iniciando API..." -ForegroundColor Green
Write-Host "🌐 URL: http://localhost:5000" -ForegroundColor Green
Write-Host "📡 Endpoints:" -ForegroundColor Green
Write-Host "   GET  /health     - Estado del servicio" -ForegroundColor Gray
Write-Host "   GET  /features   - Características" -ForegroundColor Gray
Write-Host "   POST /predict    - Realizar predicción" -ForegroundColor Gray
Write-Host "   GET/POST /test   - Prueba" -ForegroundColor Gray
Write-Host "`n🛑 Presiona CTRL+C para detener" -ForegroundColor Yellow

# Ejecutar la API
python ml_diabetes_simple.py