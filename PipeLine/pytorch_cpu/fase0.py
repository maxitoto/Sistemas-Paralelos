import os
from PIL import Image as pil
import numpy as np
import torch

def procesar(imagen_path, config):
    if not os.path.exists(imagen_path):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    device = torch.device("cpu")
    imagen_ram = pil.open(imagen_path).convert("RGB")
    
    matriz_rgb = np.asarray(imagen_ram, dtype=np.float32)
    # PyTorch exige formato (Batch, Canales, Alto, Ancho)
    matriz_rgb = np.transpose(matriz_rgb, (2, 0, 1))
    tensor_rgb = torch.tensor(matriz_rgb, device=device).unsqueeze(0)

    # 0.0 asegura que el orquestador sepa que no hay transferencia PCIe
    t_transfer_in = 0.0 
    
    return (tensor_rgb, t_transfer_in)