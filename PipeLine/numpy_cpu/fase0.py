import os
from PIL import Image as pil
import numpy as np
from time import perf_counter

def procesar(imagen_path, config):
    if not os.path.exists(imagen_path):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    try:
        # --- 1. LECTURA DE DISCO (Costo de I/O extraído de la medición) ---
        imagen_ram = pil.open(imagen_path).convert("RGB")
        
        # Llevamos la imagen a una matriz 3D en float32 lista para los cálculos
        matriz_rgb = np.asarray(imagen_ram, dtype=np.float32)

        t_transfer_in = 0

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 0: {e}")
    
    # Entregamos la matriz lista para que la Fase 1 solo haga matemática pura
    return matriz_rgb, t_transfer_in