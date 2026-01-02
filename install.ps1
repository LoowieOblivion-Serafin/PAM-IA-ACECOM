# Script de instalacion automatica para Windows
# Proyecto ACECOM - Reconstruccion de Imagenes Mentales

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "INSTALADOR AUTOMATICO - PROYECTO ACECOM" -ForegroundColor Cyan
Write-Host "Reconstruccion de Imagenes Mentales desde Actividad Cerebral" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Python 3.12
Write-Host "[1/7] Verificando Python 3.12..." -ForegroundColor Yellow
try {
    $pythonVersion = & py -3.12 --version 2>&1
    Write-Host "  OK $pythonVersion encontrado" -ForegroundColor Green
}
catch {
    Write-Host "  ERROR Python 3.12 no encontrado" -ForegroundColor Red
    Write-Host "  Instala Python 3.12 desde python.org" -ForegroundColor Red
    exit 1
}

# Clonar repositorios
Write-Host "`n[2/7] Clonando repositorios de GitHub..." -ForegroundColor Yellow

if (Test-Path "mental_img_recon-main") {
    Write-Host "  AVISO mental_img_recon-main ya existe, omitiendo..." -ForegroundColor Yellow
}
else {
    Write-Host "  Clonando mental_img_recon..." -ForegroundColor Cyan
    git clone https://github.com/nkmjm/mental_img_recon.git mental_img_recon-main
    Write-Host "  OK mental_img_recon clonado" -ForegroundColor Green
}

if (Test-Path "taming-transformers-master") {
    Write-Host "  AVISO taming-transformers-master ya existe, omitiendo..." -ForegroundColor Yellow
}
else {
    Write-Host "  Clonando taming-transformers..." -ForegroundColor Cyan
    git clone https://github.com/CompVis/taming-transformers.git taming-transformers-master
    Write-Host "  OK taming-transformers clonado" -ForegroundColor Green
}

if (Test-Path "CLIP-main") {
    Write-Host "  AVISO CLIP-main ya existe, omitiendo..." -ForegroundColor Yellow
}
else {
    Write-Host "  Clonando CLIP..." -ForegroundColor Cyan
    git clone https://github.com/openai/CLIP.git CLIP-main
    Write-Host "  OK CLIP clonado" -ForegroundColor Green
}

# Verificar archivos de compatibilidad
Write-Host "`n[3/7] Verificando archivos de compatibilidad..." -ForegroundColor Yellow

if (Test-Path "patch_taming.py") {
    Write-Host "  OK patch_taming.py presente (auto-patch para taming-transformers)" -ForegroundColor Green
}
else {
    Write-Host "  ERROR patch_taming.py NO ENCONTRADO" -ForegroundColor Red
    Write-Host "  Este archivo es necesario. Asegurate de tenerlo en el directorio." -ForegroundColor Red
}

if (Test-Path "pytorch_lightning_compat.py") {
    Write-Host "  OK pytorch_lightning_compat.py presente" -ForegroundColor Green
}
else {
    Write-Host "  ERROR pytorch_lightning_compat.py NO ENCONTRADO" -ForegroundColor Red
    Write-Host "  Este archivo es necesario. Asegurate de tenerlo en el directorio." -ForegroundColor Red
}

if (Test-Path "requirements_py312.txt") {
    Write-Host "  OK requirements_py312.txt presente" -ForegroundColor Green
}
else {
    Write-Host "  ERROR requirements_py312.txt NO ENCONTRADO" -ForegroundColor Red
    Write-Host "  Este archivo es necesario. Asegurate de tenerlo en el directorio." -ForegroundColor Red
}

# Instalar dependencias
Write-Host "`n[5/7] Instalando dependencias de Python..." -ForegroundColor Yellow
Write-Host "  Esto puede tardar varios minutos..." -ForegroundColor Cyan

py -3.12 -m pip install -r requirements_py312.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK Dependencias instaladas correctamente" -ForegroundColor Green
}
else {
    Write-Host "  AVISO Hubo problemas instalando algunas dependencias" -ForegroundColor Yellow
    Write-Host "  Revisa los mensajes arriba" -ForegroundColor Yellow
}

# Configurar config.py
Write-Host "`n[6/7] Verificando configuracion..." -ForegroundColor Yellow

$configFile = "config.py"
if (Test-Path $configFile) {
    $currentPath = (Get-Location).Path
    $content = Get-Content $configFile -Raw
    
    if ($content -match 'PROJECT_ROOT = Path\(r".*?"\)') {
        Write-Host "  AVISO PROJECT_ROOT detectado en config.py" -ForegroundColor Yellow
        Write-Host "  Ruta actual: $currentPath" -ForegroundColor Cyan
        Write-Host "  Verifica que PROJECT_ROOT en config.py coincida con tu ubicacion" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  ERROR config.py no encontrado" -ForegroundColor Red
}

# Ejecutar verificacion
Write-Host "`n[7/7] Ejecutando verificacion del setup..." -ForegroundColor Yellow
py -3.12 check_setup.py

# Resumen
Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host "INSTALACION COMPLETADA" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Fixes de compatibilidad se aplican AUTOMATICAMENTE al ejecutar" -ForegroundColor Green
Write-Host "   - patch_taming.py: Arregla torch._six en taming-transformers" -ForegroundColor White
Write-Host "   - pytorch_lightning_compat.py: Arregla PyTorch Lightning 2.x" -ForegroundColor White
Write-Host ""
Write-Host "Proximos pasos:" -ForegroundColor Green
Write-Host "  1. Extraer features.tar.gz en el directorio 'features/'" -ForegroundColor White
Write-Host "  2. Ajustar PROJECT_ROOT en config.py si es necesario (linea 22)" -ForegroundColor White
Write-Host "  3. Ejecutar: py -3.12 main_local_decoder.py" -ForegroundColor White
Write-Host ""
Write-Host "Para mas informacion, consulta README.md y SETUP.md" -ForegroundColor Cyan
