"""
===============================================================================
AUTO-PATCH PARA TAMING-TRANSFORMERS
===============================================================================

Este módulo aplica automáticamente el fix de compatibilidad torch._six a
taming-transformers SIN necesidad de modificar el repositorio clonado.

Se ejecuta ANTES de importar cualquier módulo de taming-transformers.
"""

import sys
import importlib.util

def patch_taming_transformers():
    """
    Parchea automáticamente torch._six en taming-transformers.
    
    Este fix previene el error:
    ModuleNotFoundError: No module named 'torch._six'
    
    PyTorch 1.9+ removió torch._six, pero taming-transformers aún lo usa.
    """
    # Crear módulo virtual torch._six si no existe
    if 'torch._six' not in sys.modules:
        try:
            import torch
            
            # Crear módulo virtual
            import types
            torch_six = types.ModuleType('torch._six')
            
            # Definir string_classes (usado por taming/data/utils.py)
            torch_six.string_classes = str
            
            # Inyectar en sys.modules
            sys.modules['torch._six'] = torch_six
            
            print("✓ Patch automático aplicado: torch._six creado")
            
        except ImportError:
            print("⚠ PyTorch no instalado, patch omitido")
    else:
        print("✓ torch._six ya existe (PyTorch antiguo o ya parcheado)")

# Aplicar patch al importar este módulo
patch_taming_transformers()
