import os
from PIL import Image as pil
import numpy as np
import torch
import torch.nn.functional as F
from time import perf_counter

def procesar(imagen_path, config):
    if not os.path.exists(imagen_path):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # PyTorch tarda un poco en inicializar el contexto CUDA la primera vez.
        _dummy = torch.zeros((1, 1, 256, 256), device=device)
        _k = torch.ones((1, 1, 3, 3), device=device)
        _ = F.conv2d(_dummy, _k, padding=1)
        torch.cuda.synchronize()

        imagen_ram = pil.open(imagen_path).convert("RGB")
        
        #TRANSFERENCIA (CPU -> GPU) ---
        matriz_rgb_cpu = np.asarray(imagen_ram, dtype=np.float32)
        
        # PyTorch exige formato (batch, canales, alto, ancho)
        # Transponemos de (H, W, 3) a (3, H, W) y agregamos el Batch (1, 3, H, W)
        matriz_rgb_cpu = np.transpose(matriz_rgb_cpu, (2, 0, 1))
        tensor_rgb_cpu = torch.tensor(matriz_rgb_cpu).unsqueeze(0)
        
        inicio_transfer = perf_counter()
        tensor_rgb_gpu = tensor_rgb_cpu.to(device)
        torch.cuda.synchronize()
        t_transfer_in = perf_counter() - inicio_transfer
        
    except Exception as e:
        raise ValueError(f"Falla en Fase 0: No se pudo procesar el archivo. Error: {e}")
    
    # Retornamos el tensor en la GPU, el device, y el tiempo
    return (tensor_rgb_gpu, device), t_transfer_in