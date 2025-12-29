"""
===============================================================================
SCRIPT DE DIAGNÓSTICO Y VERIFICACIÓN DEL SISTEMA
===============================================================================

Ejecuta este script ANTES de main_local_decoder.py para verificar que:
1. Todos los repositorios estén presentes
2. Las dependencias estén instaladas
3. Los datos estén disponibles (con estructura REAL del dataset)
4. El hardware sea compatible

ACTUALIZACIÓN: Verificaciones adaptadas para la estructura REAL del dataset:
- CLIP features en CLIP_ViT-B_32/lastLayer/
- VGG features en VGG19/{layer_name}/
- Ignora archivos ocultos con prefijo '._'

Uso:
    python check_setup.py
"""

import os
import sys
from pathlib import Path
import importlib

# Importar configuración
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Colores para output (compatible con Windows)
try:
    from colorama import init, Fore, Style
    init()
    GREEN = Fore.GREEN
    RED = Fore.RED
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    RESET = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = BLUE = RESET = ""

def print_header(text):
    """Imprime un encabezado con formato."""
    print("\n" + "="*70)
    print(f"{BLUE}{text}{RESET}")
    print("="*70)

def print_success(text):
    """Imprime mensaje de éxito."""
    print(f"{GREEN}✓{RESET} {text}")

def print_error(text):
    """Imprime mensaje de error."""
    print(f"{RED}✗{RESET} {text}")

def print_warning(text):
    """Imprime mensaje de advertencia."""
    print(f"{YELLOW}⚠{RESET} {text}")

def print_info(text):
    """Imprime mensaje informativo."""
    print(f"  {text}")


def check_python_version():
    """Verifica la versión de Python."""
    print_header("1. VERIFICANDO VERSIÓN DE PYTHON")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version_str} (compatible)")
        return True
    else:
        print_error(f"Python {version_str} (se requiere 3.8 o superior)")
        return False


def check_directories():
    """Verifica que existan las rutas necesarias."""
    print_header("2. VERIFICANDO ESTRUCTURA DE DIRECTORIOS")
    
    # Usar configuración centralizada
    directories = {
        "Raíz del proyecto": config.PROJECT_ROOT,
        "Repo mental_img_recon": config.REPOS['mental_img_recon'],
        "Repo taming-transformers": config.REPOS['taming_transformers'],
        "Repo CLIP": config.REPOS['clip'],
        "Dataset features": config.DATA_DIRS['features'],
        "Decoded features": config.DATA_DIRS['decoded_features'],
        "Mean features": config.DATA_DIRS['mean_features'],
    }
    
    all_exist = True
    
    for name, path in directories.items():
        if path.exists():
            print_success(f"{name}")
            print_info(f"   {path}")
        else:
            print_error(f"{name} NO ENCONTRADO")
            print_info(f"   Esperado en: {path}")
            all_exist = False
    
    return all_exist


def check_subject_data():
    """Verifica que existan datos de los sujetos con estructura REAL del dataset."""
    print_header("3. VERIFICANDO DATOS DE SUJETOS (ESTRUCTURA REAL)")
    
    decoded_dir = config.DATA_DIRS['decoded_features']
    
    if not decoded_dir.exists():
        print_error("Directorio de features decodificadas no encontrado")
        return False
    
    subjects = config.PROCESSING_CONFIG['subjects']
    all_ok = True
    
    for subject_id in subjects:
        subject_dir = decoded_dir / subject_id
        
        if not subject_dir.exists():
            print_error(f"Sujeto {subject_id}: Directorio no encontrado")
            all_ok = False
            continue
        
        print_info(f"\nSujeto {subject_id}:")
        
        # ====================================================================
        # Verificar CLIP features (CLIP_ViT-B_32/lastLayer/)
        # ====================================================================
        clip_dir = config.get_clip_path(subject_id)
        
        if clip_dir.exists():
            all_files = [f.name for f in clip_dir.iterdir() if f.is_file()]
            valid_files = config.filter_valid_files(
                all_files,
                ignore_hidden=config.PROCESSING_CONFIG['ignore_hidden_files'],
                required_substring=config.PROCESSING_CONFIG['required_substring']
            )
            valid_pkl = [f for f in valid_files if f.endswith('.pkl')]
            
            if valid_pkl:
                print_success(f"  CLIP features: {len(valid_pkl)} archivos válidos")
                hidden_count = len([f for f in all_files if f.startswith('._')])
                if hidden_count > 0:
                    print_info(f"    (ignorados {hidden_count} archivos ocultos con '._')")
            else:
                print_warning(f"  CLIP features: Sin archivos válidos")
                all_ok = False
        else:
            print_error(f"  CLIP features: Directorio no encontrado")
            print_info(f"    Esperado: {clip_dir}")
            all_ok = False
        
        # ====================================================================
        # Verificar VGG features (VGG19/{layer_name}/)
        # ====================================================================
        vgg_active_layers = config.MODEL_CONFIG['vgg_active_layers']
        vgg_ok = True
        
        for layer_name in vgg_active_layers:
            vgg_layer_dir = config.get_vgg_path(subject_id, layer_name)
            
            if vgg_layer_dir.exists():
                all_files = [f.name for f in vgg_layer_dir.iterdir() if f.is_file()]
                valid_files = config.filter_valid_files(
                    all_files,
                    ignore_hidden=config.PROCESSING_CONFIG['ignore_hidden_files'],
                    required_substring=config.PROCESSING_CONFIG['required_substring']
                )
                valid_pkl = [f for f in valid_files if f.endswith('.pkl')]
                
                if valid_pkl:
                    print_success(f"  VGG {layer_name}: {len(valid_pkl)} archivos")
                else:
                    print_warning(f"  VGG {layer_name}: Sin archivos válidos")
                    vgg_ok = False
            else:
                print_error(f"  VGG {layer_name}: Directorio no encontrado")
                vgg_ok = False
        
        if not vgg_ok:
            all_ok = False
    
    return all_ok


def check_dependencies():
    """Verifica que las dependencias estén instaladas."""
    print_header("4. VERIFICANDO DEPENDENCIAS DE PYTHON")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'omegaconf': 'OmegaConf',
        'tqdm': 'tqdm'
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            # Obtener versión si es posible
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, '__version__', 'desconocida')
                print_success(f"{name} (v{version})")
            except:
                print_success(f"{name}")
        except ImportError:
            print_error(f"{name} NO INSTALADO")
            all_installed = False
    
    return all_installed


def check_pytorch_gpu():
    """Verifica si PyTorch puede usar GPU."""
    print_header("5. VERIFICANDO SOPORTE DE GPU")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print_success(f"CUDA disponible: {gpu_count} GPU(s) detectada(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print_info(f"   GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            
            return True
        else:
            print_warning("CUDA no disponible (se usará CPU)")
            print_info("   Para usar GPU, instala PyTorch con CUDA:")
            print_info("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False
    
    except ImportError:
        print_error("PyTorch no está instalado")
        return False


def check_disk_space():
    """Verifica el espacio en disco disponible."""
    print_header("6. VERIFICANDO ESPACIO EN DISCO")
    
    try:
        import shutil
        
        project_root = config.PROJECT_ROOT
        
        if project_root.exists():
            # En Windows, usar la unidad de la ruta (ej: C:\)
            if os.name == 'nt':  # Windows
                drive = str(project_root.drive) + '\\'
            else:  # Linux/Mac
                drive = '/'
            
            total, used, free = shutil.disk_usage(drive)
            
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            print_info(f"Unidad: {drive}")
            print_info(f"Total: {total_gb:.2f} GB")
            print_info(f"Libre: {free_gb:.2f} GB")
            
            # Necesitamos ~10GB para checkpoints y resultados
            if free_gb > 10:
                print_success("Espacio suficiente disponible")
                return True
            else:
                print_warning("Espacio en disco limitado (<10GB)")
                return False
        else:
            print_warning("No se pudo verificar el espacio en disco")
            return True
    
    except Exception as e:
        print_warning(f"Error verificando espacio: {e}")
        return True


def check_vqgan_checkpoints():
    """Verifica si los checkpoints de VQGAN ya están descargados."""
    print_header("7. VERIFICANDO CHECKPOINTS DE MODELOS")
    
    # Usar rutas de config
    checkpoint_path = config.MODEL_PATHS['vqgan_checkpoint']
    config_path = config.MODEL_PATHS['vqgan_config']
    
    checkpoint_exists = checkpoint_path.exists()
    config_exists = config_path.exists()
    
    if checkpoint_exists:
        size_gb = checkpoint_path.stat().st_size / (1024**3)
        print_success(f"Checkpoint VQGAN encontrado ({size_gb:.2f} GB)")
    else:
        print_warning("Checkpoint VQGAN no encontrado")
        print_info("   Se descargará automáticamente (~4GB) al ejecutar el script")
    
    if config_exists:
        print_success("Configuración VQGAN encontrada")
    else:
        print_warning("Configuración VQGAN no encontrada")
        print_info("   Se descargará automáticamente al ejecutar el script")
    
    return True  # No es crítico, se descarga automáticamente


def estimate_processing_time():
    """Estima el tiempo de procesamiento."""
    print_header("8. ESTIMACIÓN DE TIEMPO DE PROCESAMIENTO")
    
    decoded_dir = config.DATA_DIRS['decoded_features']
    
    total_images = 0
    
    if decoded_dir.exists():
        for subject_id in config.PROCESSING_CONFIG['subjects']:
            # Contar archivos CLIP válidos (usamos CLIP como referencia)
            clip_dir = config.get_clip_path(subject_id)
            if clip_dir.exists():
                all_files = [f.name for f in clip_dir.iterdir() if f.is_file()]
                valid_files = config.filter_valid_files(
                    all_files,
                    ignore_hidden=config.PROCESSING_CONFIG['ignore_hidden_files'],
                    required_substring=config.PROCESSING_CONFIG['required_substring']
                )
                valid_pkl = [f for f in valid_files if f.endswith('.pkl')]
                total_images += len(valid_pkl)
    
    print_info(f"Total de imágenes a procesar: {total_images}")
    
    # Obtener configuración activa
    active_config = config.ACTIVE_CONFIG
    opt_params = config.get_optimization_params()
    num_iterations = opt_params['num_iterations']
    
    print_info(f"Configuración activa: {active_config}")
    print_info(f"Iteraciones por imagen: {num_iterations}")
    
    # Estimación con GPU (basada en iteraciones)
    time_per_image_gpu = (num_iterations / 100) * 1  # ~1 minuto por cada 100 iteraciones
    total_time_gpu = total_images * time_per_image_gpu
    
    # Estimación con CPU (4x más lento)
    time_per_image_cpu = time_per_image_gpu * 4
    total_time_cpu = total_images * time_per_image_cpu
    
    print_info(f"\nTiempo estimado:")
    print_info(f"  Con GPU: ~{total_time_gpu/60:.1f} horas")
    print_info(f"  Con CPU: ~{total_time_cpu/60:.1f} horas")
    
    print_info(f"\nRecomendaciones:")
    print_info(f"  - Para prueba rápida: cambiar ACTIVE_CONFIG='fast' en config.py")
    print_info(f"  - Para alta calidad: usar GPU y ACTIVE_CONFIG='high_quality'")
    print_info(f"  - Configuración actual: {active_config} ({num_iterations} iteraciones)")
    
    return True


def main():
    """Función principal del diagnóstico."""
    print("\n" + "="*70)
    print(f"{BLUE}DIAGNÓSTICO DEL SISTEMA - PROYECTO ACECOM{RESET}")
    print("Decodificación de Imágenes Mentales desde Actividad Cerebral")
    print("="*70)
    
    checks = [
        check_python_version(),
        check_directories(),
        check_subject_data(),
        check_dependencies(),
        check_pytorch_gpu(),
        check_disk_space(),
        check_vqgan_checkpoints(),
        estimate_processing_time()
    ]
    
    # Resumen final
    print_header("RESUMEN DEL DIAGNÓSTICO")
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print_success(f"Todos los checks pasaron ({passed}/{total})")
        print(f"\n{GREEN}✓ El sistema está listo para ejecutar el pipeline{RESET}")
        print(f"\nSiguiente paso:")
        print(f"  python main_local_decoder.py")
    elif passed >= total - 2:
        print_warning(f"Mayoría de checks pasaron ({passed}/{total})")
        print(f"\n{YELLOW}⚠ El sistema puede ejecutarse con advertencias{RESET}")
        print(f"\nRevisar los errores/advertencias arriba antes de continuar")
    else:
        print_error(f"Varios checks fallaron ({passed}/{total})")
        print(f"\n{RED}✗ El sistema requiere configuración adicional{RESET}")
        print(f"\nCorrege los errores arriba antes de ejecutar el pipeline")
    
    print("\n" + "="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
