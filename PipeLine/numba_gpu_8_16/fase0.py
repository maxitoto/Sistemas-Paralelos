from PIL import Image as pil
import numpy as np
import os
from numba import cuda
import math
from time import perf_counter  # IMPORTAMOS EL CRONÓMETRO

gx_kernel = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
gy_kernel = ((1, 2, 1), (0, 0, 0), (-1, -2, -1))

def procesar(imagen_path, config):
    if (not os.path.exists(imagen_path)):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    try:
        imagen_ram = pil.open(imagen_path)
        
        # --- WARMUP ---
        aux_matriz_rgb_cpu = np.asarray(imagen_ram, dtype=np.float32)
        aux_height, aux_width = aux_matriz_rgb_cpu.shape[:2]

        aux_matriz_rgb_gpu = cuda.to_device(aux_matriz_rgb_cpu) 
        aux_matriz_gris_gpu = cuda.device_array((aux_height, aux_width), dtype=np.uint8)
        aux_matriz_sobel_gpu = cuda.device_array((aux_height, aux_width), dtype=np.uint8)

        AUX_THREADS_PER_BLOCK = (8, 16) 
        aux_blockspergrid_y = (aux_height + AUX_THREADS_PER_BLOCK[0] - 1) // AUX_THREADS_PER_BLOCK[0]
        aux_blockspergrid_x = (aux_width + AUX_THREADS_PER_BLOCK[1] - 1) // AUX_THREADS_PER_BLOCK[1]
        aux_blockspergrid = (aux_blockspergrid_y, aux_blockspergrid_x)

        rgb_to_gray_cuda[aux_blockspergrid, AUX_THREADS_PER_BLOCK](aux_matriz_rgb_gpu, aux_matriz_gris_gpu)
        sobel_cuda[aux_blockspergrid, AUX_THREADS_PER_BLOCK](aux_matriz_gris_gpu, aux_matriz_sobel_gpu)
        cuda.synchronize()

        # --- MEDICIÓN DE TRANSFERENCIA (CPU -> GPU) ---
        matriz_rgb_cpu = np.asarray(imagen_ram, dtype=np.float32)
        height, width = matriz_rgb_cpu.shape[:2] 
        
        inicio_transfer = perf_counter()
        matriz_rgb_gpu = cuda.to_device(matriz_rgb_cpu)
        cuda.synchronize() # Sincronizamos para que termine la copia física
        t_transfer_in = perf_counter() - inicio_transfer
        
        # Reservamos el resto de memoria (no se cuenta como transferencia pura de datos)
        matriz_gris_gpu = cuda.device_array((height, width), dtype=np.uint8)
        matriz_sobel_gpu = cuda.device_array((height, width), dtype=np.uint8)
        THREADS_PER_BLOCK = (8, 16)

        blockspergrid_y = (height + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0]
        blockspergrid_x = (width + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1]
        blockspergrid = (blockspergrid_y, blockspergrid_x)
        
    except Exception as e:
        raise ValueError(f"Falla en Fase 0: No se pudo procesar el archivo. Error: {e}")
    
    # IMPORTANTE: Retornamos los datos encapsulados y, por separado, el tiempo
    datos_pipeline = (matriz_rgb_gpu, matriz_gris_gpu, matriz_sobel_gpu, height, width, blockspergrid, THREADS_PER_BLOCK)
    return datos_pipeline, t_transfer_in


@cuda.jit
def rgb_to_gray_cuda(rgb: np.ndarray, gray: np.ndarray) -> None:
    y, x = cuda.grid(2)
    if y < rgb.shape[0] and x < rgb.shape[1]:
        r = float(rgb[y, x, 0])
        g = float(rgb[y, x, 1])
        b = float(rgb[y, x, 2])
        i = int(0.299 * r + 0.587 * g + 0.114 * b)
        gray[y, x] = i


@cuda.jit
def sobel_cuda(gray: np.ndarray, out: np.ndarray) -> None:
    y, x = cuda.grid(2)
    if y > 0 and y < gray.shape[0] - 1 and x > 0 and x < gray.shape[1] - 1:
        gx = 0
        gy = 0
        for ky in range(3):
            for kx in range(3):
                p = int(gray[y + ky - 1, x + kx - 1])
                gx += p * gx_kernel[ky][kx]
                gy += p * gy_kernel[ky][kx]
        mag = int(math.sqrt(gx * gx + gy * gy))
        out[y, x] = 255 if mag > 255 else mag