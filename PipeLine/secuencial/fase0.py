import os
from PIL import Image as pil
import numpy as np

def procesar(imagen_path, config):
    if not os.path.exists(imagen_path):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    # Lectura única compartida
    imagen_ram = pil.open(imagen_path).convert("RGB")
    matriz_rgb = np.asarray(imagen_ram, dtype=np.uint8)
    alto, ancho = matriz_rgb.shape[:2]
    
    t_transfer_in = 0

    # Retornamos los datos para que Fase 1 inicie sin medir tiempos de I/O
    return (matriz_rgb, alto, ancho), t_transfer_in