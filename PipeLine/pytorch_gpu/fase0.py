import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from time import perf_counter

def procesar(imagen_path, config):
    if not os.path.exists(imagen_path):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    # Restricción estricta: Solo GPU
    device = torch.device("cuda")
    
    # --- 1. LECTURA COMPARTIDA EN CPU ---
    # Única variable compartida entre el calentamiento y la medición real
    imagen_ram = Image.open(imagen_path).convert("RGB")
    
    # =================================================================
    # --- 2. SIMULACIÓN DEL PIPELINE (WARM-UP) ---
    # Todo comienza con aux_ y NO se toma el tiempo.
    # =================================================================
    
    aux_matriz_cpu = np.asarray(imagen_ram, dtype=np.float32)
    aux_matriz_cpu = np.transpose(aux_matriz_cpu, (2, 0, 1))
    aux_tensor_cpu = torch.tensor(aux_matriz_cpu).unsqueeze(0)
    
    aux_kx_cpu = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        dtype=torch.float32
    )
    aux_ky_cpu = torch.tensor(
        [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],
        dtype=torch.float32
    )
    
    # Viaje de calentamiento a la VRAM
    aux_img_gpu = aux_tensor_cpu.to(device)
    aux_kx_gpu = aux_kx_cpu.to(device)
    aux_ky_gpu = aux_ky_cpu.to(device)
    
    # Fase 1 (Calentamiento)
    aux_r = aux_img_gpu[:, 0:1, :, :]
    aux_g = aux_img_gpu[:, 1:2, :, :]
    aux_b = aux_img_gpu[:, 2:3, :, :]
    aux_gris = 0.299 * aux_r + 0.587 * aux_g + 0.114 * aux_b
    
    # Fase 2 (Calentamiento)
    aux_gx = F.conv2d(aux_gris, aux_kx_gpu, padding=1)
    aux_gy = F.conv2d(aux_gris, aux_ky_gpu, padding=1)
    aux_sobel = torch.sqrt(aux_gx * aux_gx + aux_gy * aux_gy).clamp(0.0, 255.0)
    
    # Obligamos a la GPU a terminar toda la simulación antes de avanzar
    torch.cuda.synchronize()

    # =================================================================
    # --- 3. REINICIO, PREPARACIÓN REAL Y MEDICIÓN PURA ---
    # =================================================================
    
    # Re-creamos desde la variable compartida para aislar punteros de memoria
    matriz_rgb_cpu = np.asarray(imagen_ram, dtype=np.float32)
    matriz_rgb_cpu = np.transpose(matriz_rgb_cpu, (2, 0, 1))
    tensor_rgb_cpu = torch.tensor(matriz_rgb_cpu).unsqueeze(0)

    kernel_x_cpu = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        dtype=torch.float32
    )
    kernel_y_cpu = torch.tensor(
        [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],
        dtype=torch.float32
    )

    # --- INICIO DEL CRONÓMETRO OFICIAL ---
    inicio_transfer = perf_counter()
    
    # Transferencia física real (ahora la GPU ya está despierta y los buffers de cuDNN asignados)
    d_img_in = tensor_rgb_cpu.to(device)
    d_kernel_x = kernel_x_cpu.to(device)
    d_kernel_y = kernel_y_cpu.to(device)
    
    torch.cuda.synchronize()
    t_transfer_in = perf_counter() - inicio_transfer

    # Retornamos los tensores listos para que el orquestador inicie la Fase 1 oficial
    datos = (d_img_in, d_kernel_x, d_kernel_y, device)
    
    return datos, t_transfer_in