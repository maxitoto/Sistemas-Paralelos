import os
from PIL import Image as pil
import numpy as np

def procesar(imagen_path, config):
    if not os.path.exists(imagen_path):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    # Lectura única
    imagen_ram = pil.open(imagen_path).convert("RGB")
    matriz_rgb = np.asarray(imagen_ram, dtype=np.float32)
    alto, ancho = matriz_rgb.shape[:2]
    
    # --- CORRECCIÓN CRÍTICA ---
    # Usamos 0.0 para que el isinstance() del orquestador lo detecte como Float
    t_transfer_in = 0.0
    
    datos = (matriz_rgb, alto, ancho)
    return (datos, t_transfer_in)