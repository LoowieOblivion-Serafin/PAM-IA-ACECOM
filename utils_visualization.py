"""
===============================================================================
UTILIDADES DE VISUALIZACIÓN Y ANÁLISIS
===============================================================================

Script auxiliar con funciones para:
- Visualizar imágenes reconstruidas
- Comparar con imágenes originales (si disponibles)
- Generar gráficas de métricas
- Crear mosaicos de resultados
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from pathlib import Path


def visualize_reconstruction_grid(subject_id, output_dir, max_images=16):
    """
    Crea una cuadrícula con múltiples imágenes reconstruidas de un sujeto.
    
    Args:
        subject_id: ID del sujeto ('S01', 'S02', 'S03')
        output_dir: Directorio con las reconstrucciones
        max_images: Número máximo de imágenes a mostrar
    """
    subject_dir = os.path.join(output_dir, subject_id)
    
    if not os.path.exists(subject_dir):
        print(f"Error: No se encontró el directorio {subject_dir}")
        return
    
    # Obtener lista de imágenes
    image_files = sorted([f for f in os.listdir(subject_dir) 
                         if f.endswith('.png')])[:max_images]
    
    if not image_files:
        print(f"No se encontraron imágenes en {subject_dir}")
        return
    
    # Calcular dimensiones de la cuadrícula
    n_images = len(image_files)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    # Cargar y mostrar imágenes
    for idx, img_file in enumerate(image_files):
        row = idx // n_cols
        col = idx % n_cols
        
        img_path = os.path.join(subject_dir, img_file)
        img = Image.open(img_path)
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Imagen {idx+1}", fontsize=10)
        axes[row, col].axis('off')
    
    # Ocultar ejes vacíos
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f"Reconstrucciones - Sujeto {subject_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Guardar figura
    output_path = os.path.join(output_dir, f"{subject_id}_grid.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Cuadrícula guardada en: {output_path}")
    plt.close()


def compare_with_original(reconstructed_path, original_path, save_path=None):
    """
    Compara una imagen reconstruida con la original lado a lado.
    
    Args:
        reconstructed_path: Ruta a la imagen reconstruida
        original_path: Ruta a la imagen original
        save_path: Ruta donde guardar la comparación (opcional)
    """
    try:
        recon_img = Image.open(reconstructed_path)
        orig_img = Image.open(original_path)
        
        # Redimensionar original al tamaño de la reconstrucción
        orig_img = orig_img.resize(recon_img.size)
        
        # Crear figura de comparación
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(orig_img)
        axes[0].set_title("Imagen Original", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(recon_img)
        axes[1].set_title("Reconstrucción desde fMRI", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Comparación guardada en: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    except Exception as e:
        print(f"Error en comparación: {e}")


def plot_optimization_history(losses_clip, losses_vgg, save_path=None):
    """
    Grafica la evolución de las pérdidas durante la optimización.
    
    Args:
        losses_clip: Lista de valores de pérdida CLIP
        losses_vgg: Lista de valores de pérdida VGG
        save_path: Ruta donde guardar la gráfica (opcional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Pérdida CLIP
    axes[0].plot(losses_clip, label='CLIP Loss', color='blue', linewidth=2)
    axes[0].set_xlabel('Iteración', fontsize=11)
    axes[0].set_ylabel('Pérdida', fontsize=11)
    axes[0].set_title('Alineación Semántica (CLIP)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Pérdida VGG
    axes[1].plot(losses_vgg, label='VGG Loss', color='red', linewidth=2)
    axes[1].set_xlabel('Iteración', fontsize=11)
    axes[1].set_ylabel('Pérdida', fontsize=11)
    axes[1].set_title('Reconstrucción Espacial (VGG)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfica guardada en: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_report(output_dir):
    """
    Crea un reporte HTML con todas las reconstrucciones.
    
    Args:
        output_dir: Directorio con las imágenes reconstruidas
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Reconstrucciones - Proyecto ACECOM</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
            }
            .subject-section {
                background-color: white;
                margin: 20px 0;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .image-card {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                background-color: #fafafa;
            }
            .image-card img {
                width: 100%;
                height: auto;
                border-radius: 4px;
            }
            .image-card p {
                margin: 10px 0 0 0;
                font-size: 12px;
                color: #555;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <h1>🧠 Decodificación de Imágenes Mentales desde Actividad Cerebral</h1>
        <p style="text-align: center; color: #7f8c8d;">
            Proyecto ACECOM - Basado en Koide-Majima et al. (2024)
        </p>
    """
    
    subjects = ['S01', 'S02', 'S03']
    
    for subject_id in subjects:
        subject_dir = os.path.join(output_dir, subject_id)
        
        if not os.path.exists(subject_dir):
            continue
        
        image_files = sorted([f for f in os.listdir(subject_dir) 
                             if f.endswith('.png')])
        
        if not image_files:
            continue
        
        html_content += f"""
        <div class="subject-section">
            <h2>Sujeto {subject_id}</h2>
            <p>Total de reconstrucciones: {len(image_files)}</p>
            <div class="image-grid">
        """
        
        for img_file in image_files:
            rel_path = os.path.join(subject_id, img_file)
            html_content += f"""
                <div class="image-card">
                    <img src="{rel_path}" alt="{img_file}">
                    <p>{img_file}</p>
                </div>
            """
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Guardar HTML
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Reporte HTML creado: {report_path}")
    return report_path


def calculate_metrics(reconstructed_img, original_img):
    """
    Calcula métricas de similitud entre imágenes.
    
    Args:
        reconstructed_img: Imagen reconstruida (array numpy)
        original_img: Imagen original (array numpy)
    
    Returns:
        dict con métricas: PSNR, MSE, similitud estructural
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        # Asegurar mismo tamaño
        if reconstructed_img.shape != original_img.shape:
            original_img = np.array(Image.fromarray(original_img).resize(
                (reconstructed_img.shape[1], reconstructed_img.shape[0])
            ))
        
        # Calcular métricas
        mse = np.mean((reconstructed_img - original_img) ** 2)
        psnr_value = psnr(original_img, reconstructed_img, data_range=255)
        ssim_value = ssim(original_img, reconstructed_img, 
                         multichannel=True, channel_axis=2, data_range=255)
        
        return {
            'MSE': mse,
            'PSNR': psnr_value,
            'SSIM': ssim_value
        }
    
    except ImportError:
        print("Error: Instala scikit-image para calcular métricas")
        print("pip install scikit-image")
        return {}


if __name__ == "__main__":
    """
    Ejemplo de uso del módulo de visualización.
    """
    import sys
    
    # Ruta de output
    output_dir = r"C:\Users\ALVARO\Escritorio\Deco-EGG\ACECOM-Project\output_reconstructions"
    
    if not os.path.exists(output_dir):
        print(f"Error: Directorio no encontrado: {output_dir}")
        sys.exit(1)
    
    print("="*70)
    print("GENERANDO VISUALIZACIONES")
    print("="*70 + "\n")
    
    # Crear cuadrículas para cada sujeto
    for subject_id in ['S01', 'S02', 'S03']:
        print(f"\nProcesando {subject_id}...")
        visualize_reconstruction_grid(subject_id, output_dir, max_images=16)
    
    # Crear reporte HTML
    print("\nCreando reporte HTML...")
    create_summary_report(output_dir)
    
    print("\n" + "="*70)
    print("✓ VISUALIZACIONES COMPLETADAS")
    print("="*70)
