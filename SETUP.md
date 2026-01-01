# 🛠️ Guía de Instalación y Configuración

Esta guía detalla todos los pasos necesarios para replicar el pipeline de reconstrucción de imágenes mentales en cualquier máquina.

> **Actualización (Dic 2024)**: Este proyecto ahora incluye soporte GPU automático, CLIP augmentation de 32 crops, y algoritmo 100% alineado con el paper original para máxima calidad de reconstrucción.

## Requisitos del Sistema

### Hardware
- **RAM**: 8GB mínimo (16GB recomendado)
- **GPU**: NVIDIA con CUDA (opcional, acelera ~4x)
- **Espacio en disco**: ~15GB
  - Dataset: ~3GB
  - Modelos VQGAN: ~4GB
  - Modelos CLIP/VGG: ~1GB
  - Salida: ~1GB

### Software
- **Sistema Operativo**: Windows 10/11, Linux, o macOS
- **Python**: **3.12 (recomendado y probado)** o 3.8+
  - ⚠️ Python 3.14 puede generar warnings con NumPy
  - ✅ Python 3.12 es la versión estable recomendada
- **pip**: Actualizado (`python -m pip install --upgrade pip`)
- **GPU (Opcional pero recomendado)**:
  - NVIDIA GPU con CUDA 12.x
  - Drivers NVIDIA actualizados
  - **Beneficio**: 4x más rápido (5 min/imagen vs 20 min/imagen en CPU)

## Instalación Paso a Paso

### 1. Estructura de Directorios

Crea y organiza el proyecto en la siguiente estructura:

```
ACECOM-Project/
├── features/                    # Dataset (extraer aquí)
│   ├── decoded_features/
│   │   ├── S01/
│   │   │   ├── CLIP_ViT-B_32/lastLayer/
│   │   │   │   └── *.pkl (26 archivos)
│   │   │   └── VGG19/
│   │   │       ├── features_layer7/*.pkl
│   │   │       ├── features_layer16/*.pkl
│   │   │       ├── features_layer25/*.pkl
│   │   │       └── features_layer34/*.pkl
│   │   ├── S02/ (misma estructura)
│   │   └── S03/ (misma estructura)
│   └── meanDNNfeature/
│       ├── CLIP_ViT-B_32/lastLayer/meanFeature_.mat
│       └── VGG19/{layer_name}/meanFeature_.mat
├── mental_img_recon-main/       # Clonar repositorios
├── taming-transformers-master/
├── CLIP-main/
├── main_local_decoder.py
├── config.py
└── requirements.txt
```

### 2. Clonar Repositorios

```bash
# Repositorio principal (mental_img_recon)
git clone https://github.com/nkmjm/mental_img_recon.git mental_img_recon-main

# VQGAN (taming-transformers)
git clone https://github.com/CompVis/taming-transformers.git taming-transformers-master

# CLIP
git clone https://github.com/openai/CLIP.git CLIP-main
```

### 3. Extraer Dataset

El archivo `features.tar.gz` ya debe estar descargado del repositorio original.

```bash
# Extraer en el directorio del proyecto
tar -xzf features.tar.gz

# Verificar estructura
ls features/decoded_features/S01/CLIP_ViT-B_32/lastLayer/
# Debe mostrar ~26 archivos .pkl con nombres como: imagery__*.pkl
```

### 4. Archivos de Compatibilidad (Incluidos en el Proyecto)

El proyecto incluye **parches automáticos** que se aplican al ejecutar, sin necesidad de modificar repositorios externos:

#### ✅ `patch_taming.py` - Auto-patch para taming-transformers

Este archivo crea un módulo virtual `torch._six` **antes** de importar taming-transformers, evitando el error:
```
ModuleNotFoundError: No module named 'torch._six'
```

**Ventaja**: No necesitas modificar manualmente `taming-transformers-master/taming/data/utils.py`

**Cómo funciona**:
1. Se ejecuta al inicio de `main_local_decoder.py`
2. Crea `sys.modules['torch._six']` con `string_classes = str`
3. taming-transformers lo importa sin errores

#### ✅ `pytorch_lightning_compat.py` - Fix para PyTorch Lightning 2.x

Mapea el import path antiguo `pytorch_lightning.utilities.distributed` al nuevo módulo reorganizado en PyTorch Lightning 2.x.

**Ambos parches se aplican AUTOMÁTICAMENTE** - no requieren intervención manual.

---

### 5. Instalar Dependencias

#### Opción A: Instalación con Python 3.12 (Recomendada)

```bash
py -3.12 -m pip install -r requirements_py312.txt
```

#### Opción B: Instalación manual con GPU

**Para GPU NVIDIA (CUDA 12.1 - Recomendado)**:

```bash
# PyTorch con soporte GPU (2.5 GB)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Librerías requeridas
pip install numpy ftfy regex tqdm omegaconf scipy Pillow pytorch-lightning
```

**Para CPU solamente** (si no tienes GPU):

```bash
# PyTorch CPU-only (más ligero)
pip install torch torchvision torchaudio

# Librerías requeridas
pip install numpy ftfy regex tqdm omegaconf scipy Pillow pytorch-lightning
```

**Importante para Windows**: Usa Python 3.12:

```powershell
py -3.12 -m pip install -r requirements.txt
```

#### Verificar GPU (Importante!)

Después de instalar PyTorch, **verifica que tu GPU sea detectada**:

```bash
py -3.12 -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No detectada\"}')"
```

**Salida esperada con GPU**:
```
CUDA disponible: True
GPU: NVIDIA GeForce RTX 2070  # (o tu modelo)
```

**Si dice "False"**: Estás usando PyTorch CPU. Reinstala con la Opción B (GPU) arriba.

### 5. Configuración

El archivo `config.py` contiene toda la configuración del proyecto.

#### Ajustar Ruta del Proyecto (si es necesario)

**En Windows**, edita `config.py` línea ~10:

```python
PROJECT_ROOT = Path("C:/Users/ALVARO/Escritorio/Deco-EGG/ACECOM-Project")
```

**En Linux/Mac**, cambia a:

```python
PROJECT_ROOT = Path.home() / "projects" / "ACECOM-Project"
```

#### Configurar Modo de Ejecución

Edita `config.py` para elegir velocidad vs calidad:

```python
# Prueba rápida (200 iteraciones, ~2 min/imagen GPU)
ACTIVE_CONFIG = 'fast'

# Calidad estándar (500 iteraciones, ~5 min/imagen GPU)
ACTIVE_CONFIG = 'standard'  # ← RECOMENDADO

# Alta calidad (1000 iteraciones, ~10 min/imagen GPU)
ACTIVE_CONFIG = 'high_quality'
```

## Verificación del Setup

Antes de ejecutar el pipeline, verifica que todo esté configurado:

```bash
py -3.12 check_setup.py
```

**Salida esperada:**

```
✅ Python 3.12 detectado
✅ Directorio del proyecto encontrado
✅ Dataset features/ encontrado
✅ Sujeto S01: 26 archivos CLIP, 26×4 archivos VGG
✅ Sujeto S02: 26 archivos CLIP, 26×4 archivos VGG
✅ Sujeto S03: 26 archivos CLIP, 26×4 archivos VGG
✅ PyTorch instalado
✅ GPU CUDA disponible (opcional)

📊 Total de imágenes a procesar: 78
⏱️ Tiempo estimado (GPU): ~6.5 horas
```

## Ejecución

### Ejecutar Pipeline Completo

```bash
py -3.12 main_local_decoder.py
```

**Qué hace el script:**

1. ✅ Verifica rutas y configuración
2. ✅ Descarga checkpoints VQGAN (~4GB) si no existen
3. ✅ Carga modelos (VQGAN, CLIP, VGG19)
4. ✅ Procesa 78 imágenes (26 por sujeto × 3 sujetos)
5. ✅ Guarda resultados en `output_reconstructions/`

### Monitorear Progreso

El script muestra progreso en tiempo real:

```
[1/26] Reconstruyendo: black_+
Iter 100/500 | Loss_CLIP: 0.23 | Loss_VGG: 1.45
Iter 200/500 | Loss_CLIP: 0.18 | Loss_VGG: 0.92
...
✅ Guardado: output_reconstructions/S01/S01_black_+_reconstructed.png
```

También se guarda un log completo:

```bash
tail -f output_reconstructions/reconstruction.log
```

## Parámetros Avanzados

### Ajustar Iteraciones y Learning Rate

Edita `main_local_decoder.py` en la función `reconstruct_image()`:

```python
reconstructed_image = reconstruct_image(
    ...,
    num_iterations=500,    # Más iteraciones = mayor calidad
    lr=0.05,               # Learning rate (0.01-0.1)
    lambda_vgg=0.1         # Peso de VGG vs CLIP
)
```

### Procesar Solo Algunos Sujetos

Edita `config.py`:

```python
PROCESSING_CONFIG = {
    'subjects': ['S01'],  # Solo S01 (26 imágenes)
    # 'subjects': ['S01', 'S02', 'S03'],  # Todos (78 imágenes)
}
```

### Capas VGG Activas

Edita `config.py` para usar diferentes capas:

```python
MODEL_CONFIG = {
    'vgg_active_layers': [
        'features_layer7',   # Capas tempranas (bordes, texturas)
        'features_layer16',
        'features_layer25',
        'features_layer34'   # Capas profundas (objetos, semántica)
    ]
}
```

## Solución de Problemas

### Error: UnicodeEncodeError con símbolos ✓ ⚠

**Problema**: Errores de encoding en consola Windows con símbolos Unicode.  
**Solución**: Ejecuta esto en PowerShell **antes** de correr el script:

```powershell
$env:PYTHONUTF8=1
py -3.12 main_local_decoder.py
```

Esto fuerza a Python a usar UTF-8 en lugar de cp1252.

### Error: "Python version mismatch"

**Problema**: Python 3.14 genera warnings de Numpy.  
**Solución**: Usa Python 3.12:

```bash
py -3.12 main_local_decoder.py
```

### Error: "FileNotFoundError: features/"

**Problema**: Dataset no está en la ubicación correcta.  
**Solución**: Verifica `config.py` y asegúrate de que `PROJECT_ROOT` apunta al directorio correcto.

```bash
# Windows
cd C:\Users\ALVARO\Escritorio\Deco-EGG\ACECOM-Project
dir features\decoded_features\S01

# Linux/Mac
ls features/decoded_features/S01
```

### Error: "CUDA out of memory"

**Problema**: GPU tiene poca memoria.  
**Solución**: Forzar CPU en `main_local_decoder.py`:

```python
# Línea ~100
device = torch.device("cpu")
```

### Error: "SSL: CERTIFICATE_VERIFY_FAILED"

**Problema**: Error descargando checkpoints VQGAN.  
**Solución**: El script ya tiene un fix automático. Si persiste, descarga manualmente:

```bash
# Crear directorio
mkdir -p taming-transformers-master/logs/vqgan_imagenet_f16_1024/checkpoints

# Descargar checkpoints
wget https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1 \
  -O taming-transformers-master/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt
```

### Imágenes Borrosas o de Baja Calidad

**Solución**: Aumenta iteraciones y reduce learning rate:

```python
num_iterations = 1000
lr = 0.03
```

### Proceso Muy Lento (CPU)

**Estimaciones**:
- GPU: ~5 min/imagen → 6.5 horas total
- CPU: ~20 min/imagen → 26 horas total

**Recomendaciones**:
1. Usar GPU con CUDA instalado
2. Reducir iteraciones a 200-300 para pruebas
3. Procesar solo 1 sujeto primero

## Validación de Resultados

### Verificar Output

```bash
ls output_reconstructions/S01/
# Debe mostrar 26 archivos .png

# Ver una imagen
# Windows: start output_reconstructions/S01/S01_black_+_reconstructed.png
# Linux: xdg-open output_reconstructions/S01/S01_black_+_reconstructed.png
# Mac: open output_reconstructions/S01/S01_black_+_reconstructed.png
```

### Generar Reporte Visual

```bash
py -3.12 utils_visualization.py
```

Esto crea:
- `S01_grid.png`, `S02_grid.png`, `S03_grid.png` - Cuadrículas de todas las imágenes
- `report.html` - Reporte interactivo (abre en navegador)

## Timepo Estimado de Ejecución

| Configuración | Iteraciones | GPU (CUDA) | CPU |
|---------------|-------------|------------|-----|
| fast | 200 | ~2.6 horas | ~10.4 horas |
| standard | 500 | ~6.5 horas | ~26 horas |
| high_quality | 1000 | ~13 horas | ~52 horas |

*Tiempos para 78 imágenes (3 sujetos × 26 imágenes)*

## Próximos Pasos

1. ✅ Ejecutar `check_setup.py` para verificar
2. ✅ Ejecutar `main_local_decoder.py` para reconstruir
3. ✅ Revisar resultados en `output_reconstructions/`
4. ✅ Generar visualizaciones con `utils_visualization.py`

## Referencias Técnicas

- **Paper**: Koide-Majima et al. (2024) - Mental image reconstruction from human brain activity
- **VQGAN**: Esser et al. (2021) - Taming Transformers
- **CLIP**: Radford et al. (2021) - Learning Transferable Visual Models
- **VGG**: Simonyan & Zisserman (2014) - Very Deep Convolutional Networks

---

**¿Problemas?** Consulta el log detallado en `output_reconstructions/reconstruction.log` o revisa el código en `main_local_decoder.py` (está comentado extensivamente).
