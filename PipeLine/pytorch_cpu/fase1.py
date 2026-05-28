import os
from PIL import Image as pil
import numpy as np
import torch

def procesar(imagen_path, config):
    if not os.path.exists(imagen_path):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    try:
        device = torch.device("cpu")
        imagen_ram = pil.open(imagen_path).convert("RGB")
        
        # Pasamos a matriz numpy temporalmente
        matriz_rgb = np.asarray(imagen_ram, dtype=np.float32)

        # PyTorch exige formato (Batch, Canales, Alto, Ancho)
        # Transponemos de (H, W, 3) a (3, H, W)
        matriz_rgb = np.transpose(matriz_rgb, (2, 0, 1))
        tensor_rgb = torch.tensor(matriz_rgb, device=device).unsqueeze(0)

        # Separamos los canales
        r = tensor_rgb[:, 0:1, :, :]
        g = tensor_rgb[:, 1:2, :, :]
        b = tensor_rgb[:, 2:3, :, :]

        # Calculamosvectorial de luminancia
        tensor_gris = 0.299 * r + 0.587 * g + 0.114 * b

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 1: {e}")
    
    # Lo retornamos como float32 para que la Fase 2 pueda operarlo sin perder precisión
    return tensor_gris