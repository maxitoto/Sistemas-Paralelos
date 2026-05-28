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
        # Forzamos estrictamente el uso cd cpu
        device = torch.device("cpu")
        
        # Warmup de librerías subyacentes openMP y mkl
        _dummy = torch.zeros((1, 1, 256, 256), device=device)
        _k = torch.ones((1, 1, 3, 3), device=device)
        _ = F.conv2d(_dummy, _k, padding=1)

        imagen_ram = pil.open(imagen_path).convert("RGB")
        
        # preparacion
        matriz_rgb_cpu = np.asarray(imagen_ram, dtype=np.float32)
        
        # PyTorch exige formato (batch, canales, alto, ancho)
        matriz_rgb_cpu = np.transpose(matriz_rgb_cpu, (2, 0, 1))
        tensor_rgb_cpu = torch.tensor(matriz_rgb_cpu).unsqueeze(0)
        
        # TRANSFERENCIA (Numpy a Tensor en RAM) ---
        inicio_transfer = perf_counter()
        tensor_rgb_device = tensor_rgb_cpu.to(device) 
        t_transfer_in = perf_counter() - inicio_transfer
        
    except Exception as e:
        raise ValueError(f"Falla en Fase 0: No se pudo procesar el archivo. Error: {e}")
    
    return (tensor_rgb_device, device), t_transfer_in