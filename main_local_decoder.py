"""
===============================================================================
SCRIPT DE DECODIFICACIÓN DE IMÁGENES MENTALES DESDE ACTIVIDAD CEREBRAL
===============================================================================

Autor: Proyecto ACECOM - Basado en Koide-Majima et al. (2024)
Referencia: "Mental image reconstruction from human brain activity: Neural 
            decoding of mental imagery via deep neural network-based Bayesian 
            estimation"

DESCRIPCIÓN:
-----------
Este script implementa el pipeline completo de decodificación de imágenes mentales
desde actividad cerebral fMRI. Utiliza tres modelos de deep learning:

1. VQGAN (Vector Quantized GAN): Generador de imágenes desde espacio latente
   - Función: Transforma vectores latentes z → imágenes RGB de alta calidad
   - Componente: Módulo 2 (Decodificador Generativo) del paper

2. CLIP (Contrastive Language-Image Pre-training): Alineación semántica
   - Función: Espacio latente multimodal que captura significado visual
   - Componente: "Puente universal" entre cerebro e imagen (Sección 5.4.2 del PDF)

3. VGG19: Red de pérdida perceptual
   - Función: Captura características visuales jerárquicas (bordes, texturas)
   - Componente: Reconstrucción espacial de detalles finos

ARQUITECTURA DEL PROYECTO (Ver PAM_IA_ACECOM_Entregable1_V5.pdf):
----------------------------------------------------------------
                    ┌─────────────────────────────────────┐
                    │   Actividad Cerebral (fMRI)        │
                    │   [Vóxeles de V1, V2, V3, V4...]   │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  MÓDULO 1: Codificador Cerebral     │
                    │  (YA EJECUTADO - features/*.pkl)    │
                    │  f_enc: fMRI → Espacio CLIP         │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  Features Decodificadas             │
                    │  - CLIP embeddings (768-d)          │
                    │  - VGG19 features (4096-d)          │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  MÓDULO 2: Decodificador Generativo │  ← ESTE SCRIPT
                    │  (Optimización Bayesiana)           │
                    │                                      │
                    │  Objetivo: Encontrar z* que minimice:│
                    │  L = L_CLIP(z) + λ·L_VGG(z)         │
                    │                                      │
                    │  Donde:                              │
                    │  - L_CLIP: Alineación semántica      │
                    │  - L_VGG: Reconstrucción espacial    │
                    │  - z: Vector latente de VQGAN        │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  Imagen Mental Reconstruida          │
                    │  I = VQGAN_decoder(z*)              │
                    └──────────────────────────────────────┘

PROCESO DE OPTIMIZACIÓN (Langevin Dynamics):
-------------------------------------------
La reconstrucción no es directa. En su lugar, optimizamos un vector latente z
en el espacio de VQGAN que, al ser decodificado, produce una imagen cuyas
características CLIP y VGG coincidan con las decodificadas del cerebro.

Matemáticamente (Ecuación del paper):
    Pr(I|Φ_VGG, Φ_CLIP) ∝ Pr(Φ_VGG|I) × Pr(Φ_CLIP|I) × Pr(I)
                           ↑              ↑              ↑
                    Likelihood      Likelihood      Prior
                    (Visual)       (Semántico)    (Natural)

Implementación:
    z_{t+1} = z_t - η·∇_z L(z_t) + √(2η)·ε
    donde ε ~ N(0, I) es ruido gaussiano (Langevin dynamics)
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECCIÓN A: GESTIÓN DE RUTAS (PATH PATCHING)
# ============================================================================
# NOTA: Configuración para Windows usando pathlib.Path (compatible multiplataforma)

# Fix automático para taming-transformers (torch._six removido en PyTorch 1.9+)
import patch_taming

# Fix PyTorch Lightning compatibility issues BEFORE importing any other modules
import pytorch_lightning_compat

# Importar configuración centralizada
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

PROJECT_ROOT = config.PROJECT_ROOT

# Añadir rutas de repositorios al path
sys.path.insert(0, str(config.REPOS['mental_img_recon']))
sys.path.insert(0, str(config.REPOS['taming_transformers']))
sys.path.insert(0, str(config.REPOS['clip']))

print(f"✓ Añadido al path: {config.REPOS['mental_img_recon']}")
print(f"✓ Añadido al path: {config.REPOS['taming_transformers']}")
print(f"✓ Añadido al path: {config.REPOS['clip']}")

# Rutas de datos y salida (usar configuración centralizada)
FEATURES_DIR = config.DATA_DIRS['features']
DECODED_FEATURES_DIR = config.DATA_DIRS['decoded_features']
MEAN_FEATURES_DIR = config.DATA_DIRS['mean_features']
OUTPUT_DIR = config.DATA_DIRS['output']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(OUTPUT_DIR / 'reconstruction.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CLIP AUGMENTATION FUNCTION (from original paper)
# ============================================================================

def create_crops(img, num_crops=32, device='cuda'):
    """
    Generate augmented crops for CLIP input (from mental_img_recon/recon_func.py).
    
    This implements the data augmentation strategy from the original paper which:
    1. Adds padding to allow flexible cropping
    2. Applies random horizontal flips and affine transformations
    3. Generates multiple random crops with varying sizes
    4. Adds gaussian noise for regularization
    
    This is CRITICAL for reconstruction quality - it provides 32x more gradient
    signal compared to using a single image.
    
    Args:
        img: Input image tensor [1, 3, H, W] in range [0, 1]
        num_crops: Number of augmented crops to generate (default: 32)
        device: 'cuda' or 'cpu'
    
    Returns:
        Augmented crops tensor [num_crops, 3, H, W]
    
    Reference:
        Line 48-79 in mental_img_recon-main/recon_func.py:createCrops()
    """
    size1 = img.shape[2]  # Height
    size2 = img.shape[3]  # Width
    
    # Random augmentation: horizontal flip + affine transform
    augTransform = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(30, (.2, .2), fill=0)
    ).to(device)
    
    noise_factor = 0.22
    p = size1 // 2  # Padding size
    
    # Pad image (e.g., 224x224 → 448x448 with p=112)
    img_padded = torch.nn.functional.pad(img, (p, p, p, p), mode='constant', value=0)
    img_padded = augTransform(img_padded)
    
    crop_set = []
    for ch in range(num_crops):
        # Random crop size (scale between 0.43x and 1.9x of original)
        gap1 = int(torch.normal(torch.tensor(1.2), torch.tensor(0.3)).clip(0.43, 1.9) * size1)
        
        # Random crop position
        offsetx = torch.randint(0, int(size1*2-gap1), (1,)).item()
        offsety = torch.randint(0, int(size1*2-gap1), (1,)).item()
        
        # Extract crop
        crop = img_padded[:, :, offsetx:offsetx+gap1, offsety:offsety+gap1]
        
        # Resize crop back to original size
        crop = torch.nn.functional.interpolate(
            crop, (size1, size2), mode='bilinear', align_corners=True)
        crop_set.append(crop)
    
    # Concatenate all crops
    img_crops = torch.cat(crop_set, 0)  # [num_crops, 3, H, W]
    
    # Add gaussian noise for regularization
    randnormal = torch.randn_like(img_crops, requires_grad=False)
    randstotal = torch.rand((img_crops.shape[0], 1, 1, 1)).to(device)
    img_crops = img_crops + noise_factor * randstotal * randnormal
    
    return img_crops


# ============================================================================
# SECCIÓN B: DESCARGA Y CARGA DE MODELOS
# ============================================================================

def download_vqgan_checkpoints():
    """
    Descarga los checkpoints de VQGAN si no existen localmente.
    
    Archivos necesarios:
    - last.ckpt: Pesos del modelo VQGAN entrenado en ImageNet
    - model.yaml: Configuración de la arquitectura
    
    Referencias:
    - Paper original VQGAN: Esser et al. (2021) "Taming Transformers"
    - URL: https://github.com/CompVis/taming-transformers
    """
    import urllib.request
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Usar rutas de config.py
    checkpoint_path = config.MODEL_PATHS['vqgan_checkpoint']
    config_path = config.MODEL_PATHS['vqgan_config']
    
    # Crear directorios si no existen
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # URLs oficiales de Heidelberg University
    checkpoint_url = "https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1"
    config_url = "https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1"
    
    # Descargar checkpoint si no existe (~4GB)
    if not checkpoint_path.exists():
        logger.info("Descargando VQGAN checkpoint (esto puede tardar varios minutos)...")
        try:
            urllib.request.urlretrieve(checkpoint_url, str(checkpoint_path))
            logger.info(f"✓ Checkpoint descargado: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error descargando checkpoint: {e}")
            raise
    else:
        logger.info(f"✓ Checkpoint ya existe: {checkpoint_path}")
    
    # Descargar configuración
    if not config_path.exists():
        logger.info("Descargando configuración de VQGAN...")
        try:
            urllib.request.urlretrieve(config_url, str(config_path))
            logger.info(f"✓ Configuración descargada: {config_path}")
        except Exception as e:
            logger.error(f"Error descargando configuración: {e}")
            raise
    else:
        logger.info(f"✓ Configuración ya existe: {config_path}")
    
    return str(checkpoint_path), str(config_path)


def load_vqgan_model(device):
    """
    Carga el modelo VQGAN desde el repositorio local taming-transformers.
    
    VQGAN es un Autoencoder Variacional con cuantización vectorial que:
    1. Comprime imágenes a un espacio latente discreto
    2. Permite generación de alta calidad mediante su decoder
    
    Retorna:
        model: Modelo VQGAN cargado en modo evaluación
    """
    try:
        from omegaconf import OmegaConf
        from taming.models.vqgan import VQModel
        
        checkpoint_path, config_path = download_vqgan_checkpoints()
        
        # Cargar configuración
        config = OmegaConf.load(config_path)
        
        # Inicializar modelo
        model = VQModel(**config.model.params)
        
        # Cargar pesos
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.to(device)
        model.eval()
        
        logger.info("✓ Modelo VQGAN cargado correctamente")
        return model
    
    except Exception as e:
        print(f"DEBUG ERROR VQGAN LOAD: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Error cargando VQGAN: {e}")
        logger.info("Intentando cargar desde mental_img_recon-main...")
        
        # Fallback: intentar desde el repo principal
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "mental_img_recon-main"))
            from utils import load_vqgan
            model = load_vqgan(device)
            logger.info("✓ VQGAN cargado desde mental_img_recon-main")
            return model
        except Exception as e2:
            logger.error(f"Error en fallback: {e2}")
            raise


def load_clip_model(device):
    """
    Carga CLIP (ViT-B/32) desde el repositorio local.
    
    CLIP (Contrastive Language-Image Pre-training) es el componente de
    ALINEACIÓN SEMÁNTICA del pipeline. Funciona como:
    - Encoder de imágenes: Vision Transformer (ViT) que mapea imágenes → R^768
    - Espacio compartido: Embeddings normalizados en una hiperesfera unitaria
    
    Conceptualmente (Sección 5.4.2 del PDF):
    "CLIP construye un espacio de embeddings compartido entre imágenes y texto
    mediante aprendizaje contrastivo... El entrenamiento fuerza a que el vector
    de una imagen y el vector de su descripción apunten en la misma dirección"
    
    Matemáticamente:
        z_image = normalize(CLIP_encoder(I))
        Similitud = cosine(z_image, z_cerebro) = z_image · z_cerebro
    """
    try:
        import clip
        
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        
        logger.info("✓ Modelo CLIP (ViT-B/32) cargado correctamente")
        logger.info(f"  - Dimensión del embedding: {model.visual.output_dim}")
        return model, preprocess
    
    except Exception as e:
        logger.error(f"Error cargando CLIP: {e}")
        raise


def load_vgg19_model(device):
    """
    Carga VGG19 pre-entrenado para pérdida perceptual.
    
    VGG19 es el componente de RECONSTRUCCIÓN ESPACIAL que captura:
    - Características jerárquicas: capas tempranas → bordes, capas profundas → objetos
    - Pérdida perceptual: mide similitud en espacio de features, no píxeles
    
    Teoría (Sección 5.2 del PDF):
    "Esta jerarquía es análoga a la organización del sistema visual ventral
    (V1 → V2 → V4 → IT), donde las señales fMRI de diferentes regiones de
    interés (ROIs) capturan niveles jerárquicos de procesamiento visual."
    
    Usamos capas: relu1_2, relu2_2, relu3_2, relu4_2, relu5_2
    """
    try:
        from torchvision import models
        
        vgg19 = models.vgg19(pretrained=True).features.to(device)
        vgg19.eval()
        
        # FIX: Desactivar inplace=True en ReLUs porque causa errores de autograd
        for module in vgg19.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        
        # Congelar parámetros (solo extracción de features)
        for param in vgg19.parameters():
            param.requires_grad = False
        
        logger.info("✓ Modelo VGG19 cargado correctamente")
        return vgg19
    
    except Exception as e:
        logger.error(f"Error cargando VGG19: {e}")
        raise


# ============================================================================
# SECCIÓN C: LECTURA DEL DATASET 'features'
# ============================================================================

def read_decoded_features(subject_id):
    """
    Lee los archivos .pkl de decoded_features para un sujeto según estructura REAL.
    
    ESTRUCTURA REAL DEL DATASET:
    decoded_features/
    ├── {subject_id}/
    │   ├── CLIP_ViT-B_32/lastLayer/*.pkl
    │   └── VGG19/{layer_name}/*.pkl
    
    Estos archivos contienen las características DECODIFICADAS del cerebro,
    es decir, el resultado del MÓDULO 1 (Codificador Cerebral):
        f_enc: señal_fMRI → embedding_CLIP/VGG
    
    Args:
        subject_id: 'S01', 'S02', o 'S03'
    
    Returns:
        dict: {image_id: {'clip': tensor, 'vgg': dict_of_layer_tensors}}
    """
    subject_dir = DECODED_FEATURES_DIR / subject_id
    
    if not subject_dir.exists():
        logger.error(f"Directorio no encontrado: {subject_dir}")
        return {}
    
    features_dict = {}
    
    # ========================================================================
    # 1. LEER CLIP FEATURES (desde CLIP_ViT-B_32/lastLayer/)
    # ========================================================================
    clip_dir = config.get_clip_path(subject_id)
    
    if not clip_dir.exists():
        logger.error(f"Directorio CLIP no encontrado: {clip_dir}")
        return {}
    
    # Obtener lista de archivos válidos (ignorar ._ y filtrar imagery__)
    all_clip_files = [f.name for f in clip_dir.iterdir() if f.is_file()]
    valid_clip_files = config.filter_valid_files(
        all_clip_files,
        ignore_hidden=config.PROCESSING_CONFIG['ignore_hidden_files'],
        required_substring=config.PROCESSING_CONFIG['required_substring']
    )
    
    logger.info(f"Procesando {len(valid_clip_files)} archivos CLIP de {subject_id}...")
    
    # Leer cada archivo CLIP
    for pkl_file in valid_clip_files:
        if not pkl_file.endswith('.pkl'):
            continue
        
        try:
            pkl_path = clip_dir / pkl_file
            
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extraer ID de la imagen del nombre del archivo
            # Formato: "imagery__n04507155_21299.pkl" → "n04507155_21299"
            image_id = pkl_file.replace('imagery__', '').replace('.pkl', '')
            
            # Convertir a tensor (puede estar con diferentes nombres)
            if isinstance(data, dict):
                clip_feat = data.get('feat', data.get('features', data))
            else:
                clip_feat = data
            
            clip_feat = torch.tensor(clip_feat, dtype=torch.float32)
            
            # Normalizar CLIP (debe estar en la hiperesfera unitaria)
            clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
            
            # Inicializar entrada en el diccionario
            features_dict[image_id] = {
                'clip': clip_feat,
                'vgg': {}  # Se llenará con múltiples capas
            }
        
        except Exception as e:
            logger.warning(f"Error leyendo CLIP {pkl_file}: {e}")
            continue
    
    # ========================================================================
    # 2. LEER VGG FEATURES (desde VGG19/{layer_name}/)
    # ========================================================================
    vgg_active_layers = config.MODEL_CONFIG['vgg_active_layers']
    
    for layer_name in vgg_active_layers:
        vgg_layer_dir = config.get_vgg_path(subject_id, layer_name)
        
        if not vgg_layer_dir.exists():
            logger.warning(f"Directorio VGG {layer_name} no encontrado: {vgg_layer_dir}")
            continue
        
        # Obtener archivos válidos de esta capa
        all_vgg_files = [f.name for f in vgg_layer_dir.iterdir() if f.is_file()]
        valid_vgg_files = config.filter_valid_files(
            all_vgg_files,
            ignore_hidden=config.PROCESSING_CONFIG['ignore_hidden_files'],
            required_substring=config.PROCESSING_CONFIG['required_substring']
        )
        
        logger.info(f"  Leyendo {len(valid_vgg_files)} archivos de capa {layer_name}...")
        
        for pkl_file in valid_vgg_files:
            if not pkl_file.endswith('.pkl'):
                continue
            
            try:
                pkl_path = vgg_layer_dir / pkl_file
                
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Extraer ID de la imagen
                image_id = pkl_file.replace('imagery__', '').replace('.pkl', '')
                
                # Solo agregar si ya existe la entrada CLIP
                if image_id not in features_dict:
                    logger.warning(f"Imagen {image_id} no encontrada en CLIP, saltando VGG")
                    continue
                
                # Convertir a tensor
                if isinstance(data, dict):
                    vgg_feat = data.get('feat', data.get('features', data))
                else:
                    vgg_feat = data
                
                vgg_feat = torch.tensor(vgg_feat, dtype=torch.float32)
                
                # Agregar features de esta capa
                features_dict[image_id]['vgg'][layer_name] = vgg_feat
            
            except Exception as e:
                logger.warning(f"Error leyendo VGG {layer_name}/{pkl_file}: {e}")
                continue
    
    # ========================================================================
    # 3. FILTRAR IMÁGENES INCOMPLETAS (que no tienen todas las capas VGG)
    # ========================================================================
    complete_features = {}
    for image_id, features in features_dict.items():
        # Verificar que tenga todas las capas VGG activas
        has_all_layers = all(
            layer_name in features['vgg'] 
            for layer_name in vgg_active_layers
        )
        
        if has_all_layers:
            complete_features[image_id] = features
        else:
            missing_layers = [
                layer for layer in vgg_active_layers 
                if layer not in features['vgg']
            ]
            logger.warning(
                f"Imagen {image_id} incompleta (faltan capas: {missing_layers}), "
                "omitiendo..."
            )
    
    logger.info(f"✓ Cargadas {len(complete_features)} features completas de {subject_id}")
    return complete_features


def read_mean_features():
    """
    Lee los archivos .mat de meanDNNfeature (características de referencia).
    
    ESTRUCTURA REAL DEL DATASET:
    meanDNNfeature/
    ├── CLIP_ViT-B_32/lastLayer/meanFeature_.mat
    └── VGG19/{layer_name}/meanFeature_.mat
    
    Estos archivos contienen los embeddings VERDADEROS de las imágenes
    originales mostradas durante el experimento. Se usan como referencia
    para validar la calidad de la reconstrucción.
    
    Returns:
        dict: {'CLIP': {...}, 'VGG19': {layer_name: {...}}}
    """
    try:
        import scipy.io as sio
        
        mean_features = {'CLIP': None, 'VGG19': {}}
        
        # ====================================================================
        # 1. LEER CLIP meanFeature
        # ====================================================================
        clip_mat_path = config.get_mean_clip_path()
        if clip_mat_path.exists():
            try:
                clip_data = sio.loadmat(str(clip_mat_path))
                # Extract tensor from .mat file (key is 'mu' for mean)
                clip_feat = torch.from_numpy(clip_data['mu']).squeeze().float()
                mean_features['CLIP'] = clip_feat
                logger.info(f"✓ Características CLIP de referencia cargadas")
                logger.info(f"   Archivo: {clip_mat_path}")
            except Exception as e:
                logger.warning(f"Error leyendo CLIP meanFeature: {e}")
        else:
            logger.warning(f"Archivo CLIP meanFeature no encontrado: {clip_mat_path}")
        
        # ====================================================================
        # 2. LEER VGG19 meanFeatures (una por capa activa)
        # ====================================================================
        vgg_active_layers = config.MODEL_CONFIG['vgg_active_layers']
        
        for layer_name in vgg_active_layers:
            vgg_mat_path = config.get_mean_vgg_path(layer_name)
            
            if vgg_mat_path.exists():
                try:
                    vgg_data = sio.loadmat(str(vgg_mat_path))
                    # Extract tensor from .mat file (key is 'mu' for mean)
                    vgg_feat = torch.from_numpy(vgg_data['mu']).squeeze().float()
                    mean_features['VGG19'][layer_name] = vgg_feat
                    logger.info(f"✓ Características VGG19/{layer_name} de referencia cargadas")
                except Exception as e:
                    logger.warning(f"Error leyendo VGG19/{layer_name} meanFeature: {e}")
            else:
                logger.warning(f"Archivo VGG19/{layer_name} meanFeature no encontrado")
        
        return mean_features
    
    except Exception as e:
        logger.warning(f"Error cargando features de referencia: {e}")
        return {'CLIP': None, 'VGG19': {}}


# ============================================================================
# SECCIÓN D: FUNCIONES DE PÉRDIDA
# ============================================================================

def clip_loss(vqgan_output, target_clip_features, clip_model, mean_clip_feature, device):
    """
    Calcula pérdida CLIP usando augmentación de 32 crops y correlación.
    
    **IMPLEMENTACIÓN PAPER-ACCURATE** (mental_img_recon/recon_func.py:compute_loss_CLIP)
    
    Diferencias críticas vs implementación anterior:
    1. Augmentación: Genera 32 crops aleatorios con transforms → 32x más señal de gradiente
    2. Normalización CLIP-específica: mean=[0.4814, 0.4578, 0.4082] (NO ImageNet estándar)
    3. Resta mean feature: Centra features antes del cálculo → elimina bias
    4. Correlación: Usa cosine de features centradas (más robusto que MSE en alta dim)
    
    Fórmula (línea 143-144 en recon_func.py):
        loss = -cos_sim(x1 - mean(x1), x2 - mean(x2))
        donde x1, x2 ya tienen mean_clip_feature restado
    
    Args:
        vqgan_output: Imagen generada por VQGAN [-1, 1] tensor [1, 3, H, W]
        target_clip_features: Features CLIP del cerebro tensor [512] o [1, 512]
        clip_model: Modelo CLIP pre-entrenado
        mean_clip_feature: Feature CLIP promedio a restar tensor [512]
        device: 'cuda' o 'cpu'
    
    Returns:
        Tensor escalar: pérdida de correlación negativa
    
    Reference:
        Lines 117-159 in mental_img_recon-main/recon_func.py:compute_loss_CLIP()
    """
    # Normalizar imagen al rango [0, 1]
    vqgan_norm = (vqgan_output + 1.0) * 0.5
    vqgan_norm = torch.clamp(vqgan_norm, 0, 1)
    
    # Resize to 224x224 if needed
    if vqgan_norm.shape[2] != 224 or vqgan_norm.shape[3] != 224:
        vqgan_resized = torch.nn.functional.interpolate(
            vqgan_norm, (224, 224), mode='bilinear', align_corners=True)
    else:
        vqgan_resized = vqgan_norm
    
    # CRITICAL: Use CLIP-specific normalization (NOT ImageNet standard!)
    # These values come from convertVQGANoutputIntoCLIPinput() line 23 in recon_func.py
    normalize = transforms.Normalize(
        mean=[0.4814, 0.4578, 0.4082],
        std=[0.2686, 0.2613, 0.2757]
    )
    vqgan_normalized = normalize(vqgan_resized)
    
    # Generate 32 augmented crops (CRITICAL for reconstruction quality!)
    img_crops = create_crops(vqgan_normalized, num_crops=32, device=device)
    
    # Encode all crops with CLIP
    with torch.enable_grad():  # Need gradients!
        clip_features = clip_model.encode_image(img_crops)  # [32, 512]
        clip_features = clip_features.reshape(clip_features.shape[0], -1)
    
    # Reshape inputs if needed
    if target_clip_features.dim() == 1:
        target_clip_features = target_clip_features.unsqueeze(0)  # [1, 512]
    if mean_clip_feature.dim() == 1:
        mean_clip_feature = mean_clip_feature.unsqueeze(0)  # [1, 512]
    
    # Subtract mean feature (bias removal)
    x1 = clip_features - mean_clip_feature  # [32, 512]
    x2 = target_clip_features - mean_clip_feature  # [1, 512]
    
    # Correlation loss (lines 143-144 in recon_func.py)
    # cos_sim(x1 - mean(x1), x2 - mean(x2))
    cosSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = -cosSimilarity(
        x1 - x1.mean(dim=1, keepdim=True),
        x2 - x2.mean(dim=1, keepdim=True)
    ).mean()
    
    return loss


def vgg_perceptual_loss(vqgan_output, target_vgg_features, mean_vgg_features, vgg_model, device):
    """
    Calcula pérdida VGG perceptual usando mean subtraction y correlación.
    
    **IMPLEMENTACIÓN PAPER-ACCURATE** (mental_img_recon/recon_func.py:compute_loss_VGG)
    
    Diferencias críticas vs implementación anterior:
    1. Resta mean feature: Elimina bias de cada capa antes del cálculo
    2. Correlación: Usa cosine de features centradas (más robusto que MSE)
    3. Flatten consistente: Asegura dimensiones compatibles
    
    Fórmula (línea 189-190 en recon_func.py):
        loss = -cos_sim(x1 - mean(x1), x2 - mean(x2)) 
        donde x1, x2 ya tienen meanVGGfeature restado
    
    Args:
        vqgan_output: Imagen generada [-1, 1] tensor [1, 3, H, W]
        target_vgg_features: Dict {layer_name: features} del cerebro
        mean_vgg_features: Dict {layer_name: mean_features} para restar
        vgg_model: Modelo VGG19
        device: 'cuda' o 'cpu'
    
    Returns:
        Tensor escalar: pérdida ponderada de todas las capas
    
    Reference:
        Lines 162-205 in mental_img_recon-main/recon_func.py:compute_loss_VGG()
    """
    if not isinstance(target_vgg_features, dict):
        logger.warning("target_vgg_features no es un dict, retornando 0")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Normalizar imagen al rango [0, 1]
    vqgan_norm = (vqgan_output + 1.0) * 0.5
    vqgan_norm = torch.clamp(vqgan_norm, 0, 1)
    
    # Resize to 224x224 (VGG standard)
    if vqgan_norm.shape[-1] != 224:
        vqgan_resized = torch.nn.functional.interpolate(
            vqgan_norm, size=(224, 224), mode='bicubic', align_corners=False)
    else:
        vqgan_resized = vqgan_norm
    
    # ImageNet normalization (VGG was trained on ImageNet)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    vqgan_normalized = normalize(vqgan_resized)
    
    # Get layer weights and indices
    layer_weights = config.MODEL_CONFIG['vgg_layer_weights']
    target_layers_indices = {}
    for key in target_vgg_features.keys():
        if key.startswith('features_layer'):
            try:
                idx = int(key.replace('features_layer', ''))
                target_layers_indices[idx] = key
            except ValueError:
                continue
    
    if not target_layers_indices:
        return torch.tensor(0.0, device=device)
    
    total_loss = torch.tensor(0.0, device=device)
    max_layer = max(target_layers_indices.keys())
    
    # Forward pass through VGG
    x = vqgan_normalized
    cosSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    for i, layer in enumerate(vgg_model):
        x = layer(x)
        
        if i in target_layers_indices:
            layer_name = target_layers_indices[i]
            target_feat = target_vgg_features[layer_name].to(device)
            mean_feat = mean_vgg_features[layer_name].to(device)
            weight = layer_weights.get(layer_name, 0.25)
            
            # Flatten features to [1, num_features]
            x_flat = x.reshape(x.shape[0], -1)  # [1, C*H*W]
            
            # Ensure target and mean are also 2D
            if target_feat.dim() == 1:
                target_flat = target_feat.unsqueeze(0)  # [1, features]
            else:
                target_flat = target_feat.reshape(1, -1)
            
            if mean_feat.dim() == 1:
                mean_flat = mean_feat.unsqueeze(0)  # [1, features]
            else:
                mean_flat = mean_feat.reshape(1, -1)
            
            # Subtract mean features (bias removal)
            x1 = x_flat - mean_flat  # [1, features]
            x2 = target_flat - mean_flat  # [1, features]
            
            # Correlation loss (lines 189-190 in recon_func.py)
            layer_loss = -cosSimilarity(
                x1 - x1.mean(dim=1, keepdim=True),
                x2 - x2.mean(dim=1, keepdim=True)
            ).mean()
            
            total_loss += weight * layer_loss
        
        if i >= max_layer:
            break
    
    return total_loss


# ============================================================================
# SECCIÓN E: BUCLE DE RECONSTRUCCIÓN (OPTIMIZACIÓN)
# ============================================================================

def reconstruct_image(
    target_clip_features,
    target_vgg_features,
    mean_clip_feature,
    mean_vgg_features,
    vqgan_model,
    clip_model,
    vgg_model,
    device,
    num_iterations=500,
    lr=0.05,
    lambda_vgg=0.1
):
    """
    Optimiza el vector latente z para reconstruir una imagen mental.
    
    ALGORITMO CENTRAL DEL PROYECTO - MÓDULO 2 (Decodificador Generativo)
    =====================================================================
    
    Objetivo (Ecuación del paper):
        z* = argmin_z [ L_CLIP(z) + λ·L_VGG(z) ]
    
    Donde:
    - L_CLIP(z): Pérdida de alineación semántica (¿la imagen tiene el significado correcto?)
    - L_VGG(z): Pérdida de reconstrucción espacial (¿la imagen tiene los detalles correctos?)
    - λ: Balance entre semántica y estructura (típicamente 0.1-1.0)
    
    Método de optimización: Langevin Dynamics
    -----------------------------------------
    En lugar de usar solo gradientes (como Adam), añadimos ruido estocástico:
    
        z_{t+1} = z_t - η·∇_z L(z_t) + √(2η)·ε
                  ↑      ↑             ↑
              posición  gradiente    ruido gaussiano
              actual    (fuerza)     (exploración)
    
    Ventajas:
    1. Evita mínimos locales mediante exploración estocástica
    2. Implementa "Bayesian estimation" del título del paper
    3. Permite generar múltiples soluciones plausibles
    
    Args:
        target_clip_features: Embedding CLIP del cerebro (768-d)
        target_vgg_features: Features VGG del cerebro (4096-d)
        vqgan_model: Modelo VQGAN para decodificar z → imagen
        clip_model: Modelo CLIP para pérdida semántica
        vgg_model: Modelo VGG19 para pérdida perceptual
        device: 'cuda' o 'cpu'
        num_iterations: Número de pasos de optimización (500-1000 típicamente)
        lr: Learning rate (0.05-0.1 típicamente)
        lambda_vgg: Peso de la pérdida VGG vs CLIP
    
    Returns:
        Imagen reconstruida como array numpy [H, W, 3]
    """
    logger.info("="*60)
    logger.info("INICIANDO RECONSTRUCCIÓN DE IMAGEN MENTAL")
    logger.info("="*60)
    
    # Inicializar vector latente z aleatoriamente
    # Dimensión: [1, 256, H/16, W/16] para VQGAN f=16
    latent_size = 16  # Para imágenes 256x256 con factor 16
    z = torch.randn(1, 256, latent_size, latent_size, device=device, requires_grad=True)
    
    # Optimizador: Adam con Langevin dynamics implementado manualmente
    optimizer = optim.Adam([z], lr=lr)
    
    # Mover features objetivo al device
    target_clip_features = target_clip_features.to(device)
    
    # Barra de progreso
    pbar = tqdm(range(num_iterations), desc="Optimizando vector latente")
    
    for iteration in pbar:
        optimizer.zero_grad()
        
        # FORWARD PASS: Decodificar z → imagen
        with torch.enable_grad():
            reconstructed_image = vqgan_model.decode(z)
        
        # CALCULAR PÉRDIDAS
        # 1. Alineación Semántica (CLIP) - con augmentación y mean subtraction
        loss_clip = clip_loss(reconstructed_image, target_clip_features, clip_model, mean_clip_feature, device)
        
        # 2. Reconstrucción Espacial (VGG) - con mean subtraction y correlación
        loss_vgg = vgg_perceptual_loss(reconstructed_image, target_vgg_features, mean_vgg_features, vgg_model, device)
        
        # 3. Pérdida Total
        total_loss = loss_clip + lambda_vgg * loss_vgg
        
        # BACKWARD PASS
        total_loss.backward()
        optimizer.step()
        
        # Implementar ruido de Langevin (cada N iteraciones)
        if iteration % 10 == 0:
            with torch.no_grad():
                noise = torch.randn_like(z) * np.sqrt(2 * lr)
                z.add_(noise)
        
        # Logging
        if iteration % 50 == 0:
            logger.info(f"Iter {iteration}/{num_iterations} | "
                       f"Loss_CLIP: {loss_clip.item():.4f} | "
                       f"Loss_VGG: {loss_vgg.item():.4f} | "
                       f"Total: {total_loss.item():.4f}")
        
        pbar.set_postfix({
            'CLIP': f'{loss_clip.item():.3f}',
            'VGG': f'{loss_vgg.item():.3f}'
        })
    
    # DECODIFICACIÓN FINAL
    with torch.no_grad():
        final_image = vqgan_model.decode(z)
        final_image = (final_image + 1) / 2  # [-1, 1] → [0, 1]
        final_image = torch.clamp(final_image, 0, 1)
        
        # Convertir a numpy array [H, W, 3]
        final_image_np = final_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        final_image_np = (final_image_np * 255).astype(np.uint8)
    
    logger.info("✓ Reconstrucción completada")
    return final_image_np


# ============================================================================
# SECCIÓN F: MAIN LOOP - PROCESAMIENTO DE TODOS LOS SUJETOS
# ============================================================================

def main():
    """
    Función principal que orquesta todo el pipeline de decodificación.
    
    Pipeline:
    1. Configurar device (GPU si disponible)
    2. Cargar modelos (VQGAN, CLIP, VGG19)
    3. Para cada sujeto (S01, S02, S03):
        a. Leer features decodificadas
        b. Para cada imagen:
            - Optimizar vector latente z
            - Generar imagen reconstruida
            - Guardar en output_reconstructions/
    """
    logger.info("\n" + "="*70)
    logger.info("PIPELINE DE DECODIFICACIÓN DE IMÁGENES MENTALES")
    logger.info("Proyecto ACECOM - Basado en Koide-Majima et al. (2024)")
    logger.info("="*70 + "\n")
    
    # ========================================================================
    # 1. CONFIGURACIÓN DEL DEVICE
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠ GPU no disponible. El procesamiento será lento.")
    
    # ========================================================================
    # 2. CARGAR MODELOS
    # ========================================================================
    logger.info("\n" + "-"*70)
    logger.info("CARGANDO MODELOS DE DEEP LEARNING")
    logger.info("-"*70)
    
    try:
        # VQGAN: Generador de imágenes
        logger.info("\n[1/3] Cargando VQGAN...")
        vqgan_model = load_vqgan_model(device)
        
        # CLIP: Alineación semántica
        logger.info("\n[2/3] Cargando CLIP...")
        clip_model, clip_preprocess = load_clip_model(device)
        
        # VGG19: Pérdida perceptual
        logger.info("\n[3/3] Cargando VGG19...")
        vgg_model = load_vgg19_model(device)
        
        logger.info("\n✓ Todos los modelos cargados exitosamente\n")
    
    except Exception as e:
        logger.error(f"Error fatal cargando modelos: {e}")
        return
    
    # ========================================================================
    # 2.5 CARGAR MEAN FEATURES (CRÍTICO PARA RECONSTRUCTION QUALITY!)
    # ========================================================================
    logger.info("\n" + "-"*70)
    logger.info("CARGANDO MEAN FEATURES")
    logger.info("-"*70)
    
    try:
        mean_features = read_mean_features()
        # Fix: Use uppercase keys 'CLIP' and 'VGG19' as returned by read_mean_features()
        mean_clip_feature = mean_features['CLIP'].to(device)
        mean_vgg_features = {k: v.to(device) for k, v in mean_features['VGG19'].items()}
        logger.info(f"✓ Mean CLIP feature cargado: shape {mean_clip_feature.shape}")
        logger.info(f"✓ Mean VGG features cargados: {len(mean_vgg_features)} capas")
    except Exception as e:
        logger.error(f"Error cargando mean features: {e}")
        logger.error("Las mean features son NECESARIAS para la reconstrucción correcta.")
        return
    
    # ========================================================================
    # 3. PROCESAR CADA SUJETO
    # ========================================================================
    subjects = ['S01', 'S02', 'S03']
    
    for subject_id in subjects:
        logger.info("\n" + "="*70)
        logger.info(f"PROCESANDO SUJETO: {subject_id}")
        logger.info("="*70 + "\n")
        
        # Leer features decodificadas del sujeto
        features_dict = read_decoded_features(subject_id)
        
        if not features_dict:
            logger.warning(f"No se encontraron features para {subject_id}. Saltando...")
            continue
        
        # Crear directorio de salida para el sujeto
        subject_output_dir = OUTPUT_DIR / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Procesar cada imagen
        total_images = len(features_dict)
        logger.info(f"Total de imágenes a reconstruir: {total_images}\n")
        
        for idx, (image_id, features) in enumerate(features_dict.items(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Imagen {idx}/{total_images}: {image_id}")
            logger.info(f"{'='*60}")
            
            try:
                # Reconstruir imagen (paper-accurate algorithm)
                reconstructed_image = reconstruct_image(
                    target_clip_features=features['clip'],
                    target_vgg_features=features['vgg'],
                    mean_clip_feature=mean_clip_feature,
                    mean_vgg_features=mean_vgg_features,
                    vqgan_model=vqgan_model,
                    clip_model=clip_model,
                    vgg_model=vgg_model,
                    device=device,
                    num_iterations=500,  # Ajustar según calidad deseada
                    lr=0.05,
                    lambda_vgg=0.1
                )
                
                # Guardar imagen
                output_filename = f"{subject_id}_{image_id}_reconstructed.png"
                output_path = subject_output_dir / output_filename
                
                img = Image.fromarray(reconstructed_image)
                img.save(str(output_path))
                
                logger.info(f"✓ Imagen guardada: {output_path}")
            
            except Exception as e:
                logger.error(f"Error reconstruyendo {image_id}: {e}")
                continue
    
    # ========================================================================
    # 4. RESUMEN FINAL
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETADO")
    logger.info("="*70)
    logger.info(f"\nImágenes reconstruidas guardadas en: {OUTPUT_DIR}")
    logger.info("\nNOTA IMPORTANTE:")
    logger.info("Este localhost se refiere al localhost de la computadora que ")
    logger.info("ejecuta el script, no tu máquina local. Para acceder a las ")
    logger.info("imágenes localmente, debes descargarlas o ejecutar el script ")
    logger.info("en tu propia máquina.\n")


if __name__ == "__main__":
    """
    Punto de entrada del script.
    
    Para ejecutar:
        python main_local_decoder.py
    
    Requisitos:
    - Python 3.8+
    - PyTorch 1.10+
    - CUDA (opcional pero recomendado)
    - 8GB+ RAM (16GB recomendado)
    - ~10GB espacio en disco para checkpoints
    
    Estructura de salida:
        output_reconstructions/
        ├── S01/
        │   ├── S01_n04507155_21299_reconstructed.png
        │   ├── S01_n03788195_15632_reconstructed.png
        │   └── ...
        ├── S02/
        │   └── ...
        └── S03/
            └── ...
    """
    main()
