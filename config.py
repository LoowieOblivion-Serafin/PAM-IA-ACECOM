"""
===============================================================================
ARCHIVO DE CONFIGURACIÓN DEL PROYECTO
===============================================================================

Este archivo centraliza todos los parámetros configurables del pipeline de
decodificación. Modificar estos valores permite adaptar el comportamiento del
sistema sin editar el código principal.

Para usuarios avanzados: Puedes crear múltiples configuraciones para diferentes
experimentos (e.g., config_high_quality.py, config_fast.py)
"""

from pathlib import Path

# ============================================================================
# RUTAS DEL PROYECTO (CONFIGURADO PARA WINDOWS)
# ============================================================================

# Ruta raíz del proyecto en Windows
# NOTA: Usando pathlib.Path para compatibilidad multiplataforma
PROJECT_ROOT = Path(r"C:\Users\ALVARO\Escritorio\Deco-EGG\ACECOM-Project")

# Repositorios de código (usando operador / de pathlib)
REPOS = {
    'mental_img_recon': PROJECT_ROOT / "mental_img_recon-main",
    'taming_transformers': PROJECT_ROOT / "taming-transformers-master",
    'clip': PROJECT_ROOT / "CLIP-main"
}

# Directorios de datos (usando operador / de pathlib)
DATA_DIRS = {
    'features': PROJECT_ROOT / "features",
    'decoded_features': PROJECT_ROOT / "features" / "decoded_features",
    'mean_features': PROJECT_ROOT / "features" / "meanDNNfeature",
    'output': PROJECT_ROOT / "output_reconstructions"
}

# Checkpoints de modelos (usando operador / de pathlib)
MODEL_PATHS = {
    'vqgan_checkpoint': (
        REPOS['taming_transformers'] /
        "logs" / "vqgan_imagenet_f16_1024" / "checkpoints" / "last.ckpt"
    ),
    'vqgan_config': (
        REPOS['taming_transformers'] /
        "logs" / "vqgan_imagenet_f16_1024" / "configs" / "model.yaml"
    )
}

# ============================================================================
# PARÁMETROS DE MODELOS
# ============================================================================

MODEL_CONFIG = {
    # CLIP - Variante específica del dataset
    'clip_variant': 'ViT-B/32',  # CLIP_ViT-B_32 según dataset
    'clip_embedding_dim': 768,   # Dimensión del espacio latente
    'clip_layer': 'lastLayer',   # Capa específica a usar
    
    # VGG19 - Capas disponibles en el dataset REAL
    # Features layers: capas convolucionales
    'vgg_features_layers': [
        'features_layer2',   # Conv1_2
        'features_layer7',   # Conv2_2
        'features_layer16',  # Conv3_4
        'features_layer25',  # Conv4_4
        'features_layer34'   # Conv5_4
    ],
    # Classifier layers: capas fully connected
    'vgg_classifier_layers': [
        'classifier_layer0',  # FC1
        'classifier_layer3',  # FC2
        'classifier_layer6'   # FC3 (output)
    ],
    # Capas a usar en la reconstrucción (combinación de ambas)
    'vgg_active_layers': [
        'features_layer7',    # Características tempranas
        'features_layer16',   # Características medias
        'features_layer25',   # Características profundas
        'features_layer34'    # Características más profundas
    ],
    # Pesos por capa (suma = 1.0)
    'vgg_layer_weights': {
        'features_layer7': 0.25,
        'features_layer16': 0.25,
        'features_layer25': 0.25,
        'features_layer34': 0.25
    },
    
    # VQGAN
    'vqgan_latent_size': 16,  # Para imágenes 256x256 con factor 16
    'vqgan_latent_channels': 256
}

# ============================================================================
# PARÁMETROS DE OPTIMIZACIÓN (MÓDULO 2 - DECODIFICADOR)
# ============================================================================

OPTIMIZATION_CONFIG = {
    # -------------------------------------------
    # CONFIGURACIÓN RÁPIDA (para pruebas)
    # -------------------------------------------
    'fast': {
        'num_iterations': 200,
        'learning_rate': 0.1,
        'lambda_vgg': 0.1,
        'langevin_noise_interval': 20,
        'langevin_noise_scale': 0.01
    },
    
    # -------------------------------------------
    # CONFIGURACIÓN ESTÁNDAR (equilibrio calidad/velocidad)
    # -------------------------------------------
    'standard': {
        'num_iterations': 500,
        'learning_rate': 0.05,
        'lambda_vgg': 0.1,
        'langevin_noise_interval': 10,
        'langevin_noise_scale': 0.005
    },
    
    # -------------------------------------------
    # CONFIGURACIÓN DE ALTA CALIDAD (lento pero preciso)
    # -------------------------------------------
    'high_quality': {
        'num_iterations': 1000,
        'learning_rate': 0.03,
        'lambda_vgg': 0.15,
        'langevin_noise_interval': 5,
        'langevin_noise_scale': 0.003
    },
    
    # -------------------------------------------
    # CONFIGURACIÓN PARA PAPER (reproducir resultados exactos)
    # -------------------------------------------
    'paper': {
        'num_iterations': 800,
        'learning_rate': 0.05,
        'lambda_vgg': 0.1,
        'langevin_noise_interval': 10,
        'langevin_noise_scale': 0.005
    }
}

# Configuración activa (CAMBIAR AQUÍ PARA EXPERIMENTAR)
ACTIVE_CONFIG = 'standard'  # Opciones: 'fast', 'standard', 'high_quality', 'paper'

# ============================================================================
# PARÁMETROS DE PROCESAMIENTO
# ============================================================================

PROCESSING_CONFIG = {
    # Sujetos a procesar
    'subjects': ['S01', 'S02', 'S03'],  # Lista de sujetos
    
    # Filtros de imágenes
    'filter_imagery_only': True,  # Solo imágenes mentales (no percepción)
    'ignore_hidden_files': True,  # Ignorar archivos que empiezan con '._' (macOS)
    'required_substring': 'imagery__',  # Solo archivos que contengan esto
    'max_images_per_subject': None,  # None = todas, o número específico
    
    # Estructura del dataset (según estructura REAL)
    'dataset_structure': {
        'clip_path_template': 'CLIP_ViT-B_32/lastLayer',
        'vgg_path_template': 'VGG19/{layer_name}',  # {layer_name} se reemplaza dinámicamente
    },
    
    # Tamaño de imagen
    'image_size': 256,  # Resolución de salida (256x256)
    
    # Formato de guardado
    'save_format': 'png',  # Opciones: 'png', 'jpg'
    'save_quality': 95  # Para JPG (1-100)
}

# ============================================================================
# PARÁMETROS DE HARDWARE
# ============================================================================

HARDWARE_CONFIG = {
    # Device
    'force_cpu': False,  # True = forzar CPU incluso si hay GPU
    'gpu_id': 0,  # ID de GPU a usar (si hay múltiples)
    
    # Memoria
    'pin_memory': True,  # Usar memoria pinned para transferencias más rápidas
    'num_workers': 4,  # Número de workers para DataLoader (si se usa)
    
    # Precisión
    'use_fp16': False  # Usar precisión mixta (requiere GPU moderna)
}

# ============================================================================
# PARÁMETROS DE LOGGING
# ============================================================================

LOGGING_CONFIG = {
    # Nivel de detalle
    'log_level': 'INFO',  # Opciones: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    # Frecuencia de logging
    'log_every_n_iterations': 50,  # Imprimir cada N iteraciones
    
    # Archivos
    'save_log_file': True,
    'log_filename': 'reconstruction.log',
    
    # Visualización
    'show_progress_bar': True,
    'use_colored_logs': True  # Requiere coloredlogs
}

# ============================================================================
# PARÁMETROS DE EVALUACIÓN (opcional)
# ============================================================================

EVALUATION_CONFIG = {
    # Métricas
    'compute_metrics': False,  # Calcular métricas de calidad
    'metrics': ['ssim', 'psnr', 'lpips'],  # Métricas a calcular
    
    # Comparación con ground truth
    'compare_with_original': False,  # Requiere imágenes originales
    'original_images_path': None  # Ruta a imágenes originales
}

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_optimization_params():
    """Retorna los parámetros de optimización activos."""
    return OPTIMIZATION_CONFIG[ACTIVE_CONFIG]

def validate_paths():
    """Valida que las rutas críticas existan."""
    errors = []
    
    # Verificar repositorios
    for name, path in REPOS.items():
        if not path.exists():
            errors.append(f"Repositorio no encontrado: {name} en {path}")
    
    # Verificar directorios de datos
    if not DATA_DIRS['decoded_features'].exists():
        errors.append(f"Directorio de features no encontrado: {DATA_DIRS['decoded_features']}")
    
    return errors

def filter_valid_files(file_list, ignore_hidden=True, required_substring=None):
    """
    Filtra archivos válidos ignorando archivos ocultos y verificando substring.
    
    Args:
        file_list: Lista de nombres de archivos
        ignore_hidden: Si True, ignora archivos que empiezan con '._'
        required_substring: Si se especifica, solo incluye archivos que contengan este substring
    
    Returns:
        Lista de archivos válidos
    """
    valid_files = []
    
    for filename in file_list:
        # Ignorar archivos ocultos de macOS
        if ignore_hidden and filename.startswith('._'):
            continue
        
        # Verificar substring requerido
        if required_substring and required_substring not in filename:
            continue
        
        valid_files.append(filename)
    
    return valid_files

def get_clip_path(subject_id):
    """Retorna la ruta completa para archivos CLIP de un sujeto."""
    clip_template = PROCESSING_CONFIG['dataset_structure']['clip_path_template']
    return DATA_DIRS['decoded_features'] / subject_id / clip_template

def get_vgg_path(subject_id, layer_name):
    """Retorna la ruta completa para archivos VGG de un sujeto y capa específica."""
    vgg_template = PROCESSING_CONFIG['dataset_structure']['vgg_path_template']
    vgg_path = vgg_template.format(layer_name=layer_name)
    return DATA_DIRS['decoded_features'] / subject_id / vgg_path

def get_mean_clip_path():
    """Retorna la ruta completa para meanFeature de CLIP."""
    return DATA_DIRS['mean_features'] / 'CLIP_ViT-B_32' / 'lastLayer' / 'meanFeature_.mat'

def get_mean_vgg_path(layer_name):
    """Retorna la ruta completa para meanFeature de VGG de una capa específica."""
    return DATA_DIRS['mean_features'] / 'VGG19' / layer_name / 'meanFeature_.mat'

def print_config():
    """Imprime la configuración activa."""
    print("="*70)
    print("CONFIGURACIÓN DEL PROYECTO")
    print("="*70)
    print(f"\nRuta del proyecto: {PROJECT_ROOT}")
    print(f"\nConfiguración de optimización activa: {ACTIVE_CONFIG}")
    print(f"Parámetros:")
    for key, value in get_optimization_params().items():
        print(f"  - {key}: {value}")
    print(f"\nSujetos a procesar: {', '.join(PROCESSING_CONFIG['subjects'])}")
    print(f"Capas VGG activas: {', '.join(MODEL_CONFIG['vgg_active_layers'])}")
    print(f"Device: {'CPU' if HARDWARE_CONFIG['force_cpu'] else 'GPU (si disponible)'}")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    # Validar configuración
    print_config()
    
    errors = validate_paths()
    if errors:
        print("\n⚠️ ERRORES EN LA CONFIGURACIÓN:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Todas las rutas validadas correctamente")
