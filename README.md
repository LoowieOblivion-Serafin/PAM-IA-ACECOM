# 🧠 Reconstrucción de Imágenes Mentales desde Actividad Cerebral

Proyecto que convierte señales fMRI del cerebro humano en imágenes visuales reconstruidas.

## Descripción

Este proyecto implementa un sistema de decodificación neuronal que:
- **Lee señales cerebrales** (fMRI) de personas viendo imágenes
- **Extrae características visuales** mediante redes neuronales (CLIP, VGG19)
- **Reconstruye las imágenes mentales** usando un generador VQGAN optimizado

El sistema permite "ver" qué está visualizando una persona analizando solo su actividad cerebral.

## Características

✅ **Paper-Accurate Implementation** - Algoritmo validado científicamente  
✅ **CLIP Augmentation** - 32 crops aumentados para máxima calidad  
✅ **Mean Feature Subtraction** - Elimina bias para mejor convergencia  
✅ **Correlation Loss** - Más robusto que MSE en espacios de alta dimensión  
✅ Procesa datos de 3 sujetos (S01, S02, S03) con 26 imágenes cada uno  
✅ Pipeline completo end-to-end desde features cerebrales a imágenes  
✅ Descarga automática de modelos pre-entrenados (~4GB)  
✅ **Soporte GPU** (CUDA) - 4x más rápido que CPU  
✅ Configuración flexible con múltiples modos de calidad

## Diferencias con la Investigación Original

Este proyecto **reimplementa y mejora** el algoritmo del paper científico:

**Mejoras implementadas**:
- ✅ **Multiplataforma** - Funciona en Windows/Linux/Mac (original solo Linux)
- ✅ **GPU Auto-detection** - Detecta y usa CUDA automáticamente
- ✅ **Configuración Flexible** - 3 modos de calidad (fast/standard/high_quality)
- ✅ **Documentación Completa** - Guías paso a paso en español
- ✅ **Verificación Automática** - Script `check_setup.py` valida instalación
- ✅ **Logging Detallado** - Monitoreo de progreso en tiempo real

**Algoritmo idéntico al paper**:
- ✅ CLIP Augmentation (32 crops con transformaciones aleatorias)
- ✅ CLIP Normalization específica [0.4814, 0.4578, 0.4082]
- ✅ Mean Feature Subtraction para CLIP y VGG
- ✅ Correlation Loss en lugar de MSE
- ✅ Langevin Dynamics con ruido gaussiano cada 10 iteraciones

## Quick Start

```bash
# 1. Clonar repositorios necesarios
git clone https://github.com/nkmjm/mental_img_recon.git mental_img_recon-main
git clone https://github.com/CompVis/taming-transformers.git taming-transformers-master
git clone https://github.com/openai/CLIP.git CLIP-main

# 2. Instalar dependencias (Python 3.12)
py -3.12 -m pip install -r requirements_py312.txt

# 3. Configurar PROJECT_ROOT en config.py (línea 22)
# Ajustar la ruta a tu ubicación del proyecto

# 4. Verificar instalación
py -3.12 check_setup.py

# 5. Ejecutar pipeline
py -3.12 main_local_decoder.py

# 6. Ver resultados
explorer output_reconstructions  # Windows
# open output_reconstructions    # Mac
# xdg-open output_reconstructions # Linux
```

> **✅ VENTAJA**: Los fixes de compatibilidad se aplican **AUTOMÁTICAMENTE** al ejecutar:
> - `patch_taming.py` - Arregla `torch._six` en taming-transformers SIN modificar el repo
> - `pytorch_lightning_compat.py` - Arregla PyTorch Lightning 2.x
> 
> **No necesitas editar manualmente ningún archivo de repositorios externos**

Para instrucciones detalladas de instalación y configuración, consulta **[SETUP.md](SETUP.md)**.

## Estructura del Proyecto

```
ACECOM-Project/
├── features/                    # Dataset (extraído de features.tar.gz)
│   ├── decoded_features/        # Features cerebrales por sujeto
│   └── meanDNNfeature/          # Features promedio
├── mental_img_recon-main/       # Repositorio base
├── taming-transformers-master/  # Arquitectura VQGAN
├── CLIP-main/                   # Arquitectura CLIP
├── main_local_decoder.py        # Script principal
├── config.py                    # Configuración del proyecto
└── output_reconstructions/      # Imágenes generadas (se crea automáticamente)
```

## Requisitos del Sistema

- **Python**: 3.12 (estable, recomendado) o 3.8+
- **RAM**: 8GB mínimo, 16GB recomendado
- **GPU**: NVIDIA con CUDA (opcional, acelera 4x)
- **Espacio**: ~15GB (dataset + modelos)

## Scripts Principales

### `main_local_decoder.py`
Pipeline de reconstrucción completo. Lee features cerebrales y genera imágenes.

**Uso básico:**
```bash
py -3.12 main_local_decoder.py
```

**Configuración rápida/estándar/alta:**
Edita `ACTIVE_CONFIG` en `config.py` (`'fast'` / `'standard'` / `'high_quality'`)

### `utils_visualization.py`
Genera cuadrículas comparativas y reportes HTML.

```bash
py -3.12 utils_visualization.py
```

### `check_setup.py`
Verifica que todo esté configurado correctamente antes de ejecutar.

```bash
py -3.12 check_setup.py
```

## Autor

**Proyecto**: ACECOM - Decodificación de Imágenes Mentales  
**Estudiante**: Alvaro Jesus Taipe Cotrina  
**Institución**: Universidad Nacional de Ingeniería (UNI)  
**Año**: 2025

## Bibliografía e Inspiración

### Paper Científico Original

Este proyecto está **inspirado e implementa** el algoritmo descrito en:

> **Koide-Majima, N., Nishimoto, S.** (2024). "Mental image reconstruction from human brain activity: Neural decoding of mental imagery via deep neural network-based Bayesian estimation"

**Fundamento Científico**:

El paper propone un método bayesiano para reconstruir imágenes mentales:

1. **Codificador Cerebral** (f_enc): Mapea actividad fMRI → espacio de embeddings CLIP/VGG
2. **Optimización Bayesiana**: Minimiza `L = L_CLIP + λ·L_VGG` usando dinámica de Langevin
3. **Generador VQGAN**: Sintetiza imagen final desde el espacio latente optimizado

**Ecuación Central**:
```
Pr(I|Φ_VGG, Φ_CLIP) ∝ Pr(Φ_VGG|I) × Pr(Φ_CLIP|I) × Pr(I)
                       ↑              ↑              ↑
                  Likelihood      Likelihood      Prior
                  (Visual)       (Semántico)    (Natural)
```

### Repositorios Base

Este proyecto utiliza y se basa en los siguientes repositorios:

- **Implementación Original**: [nkmjm/mental_img_recon](https://github.com/nkmjm/mental_img_recon)
  - Código fuente del paper
  - Dataset de features cerebrales pre-extraídas
  
- **VQGAN**: [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
  - Esser et al. (2021) "Taming Transformers for High-Resolution Image Synthesis"
  - Generador de imágenes de alta calidad
  
- **CLIP**: [openai/CLIP](https://github.com/openai/CLIP)
  - Radford et al. (2021) "Learning Transferable Visual Models From Natural Language Supervision"
  - Espacio latente multimodal

### Dataset  
**datos/fmri**: (https://drive.google.com/uc?id=1Q7TVsVbASMqnDYfFjFzo2SV6njExu8qq)
El dataset `features.tar.gz` (NO incluido en GitHub por tamaño) contiene:

- Features cerebrales pre-extraídas de fMRI
- 3 sujetos (S01, S02, S03)
- 26 imágenes por sujeto
- Features CLIP (512-d) y VGG19 (4096-d por capa)

**Obtención**: Descarga desde el repositorio original [nkmjm/mental_img_recon](https://github.com/nkmjm/mental_img_recon)

## Licencia

Este proyecto es con fines académicos y de investigación. Los modelos pre-entrenados (VQGAN, CLIP, VGG19) mantienen sus licencias originales.

---

**🚀 Para comenzar, consulta [SETUP.md](SETUP.md) para instrucciones detalladas de instalación.**

