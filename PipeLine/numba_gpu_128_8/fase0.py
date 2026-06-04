import os
from PIL import Image as pil
import numpy as np
from numba import cuda
import math
from time import perf_counter

gx_kernel = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
gy_kernel = ((1, 2, 1), (0, 0, 0), (-1, -2, -1))

@cuda.jit
def rgb_to_gray_cuda(rgb, gray_float, gray_uint8):
    y, x = cuda.grid(2)
    if y < rgb.shape[0] and x < rgb.shape[1]:
        r = float(rgb[y, x, 0])
        g = float(rgb[y, x, 1])
        b = float(rgb[y, x, 2])
        
        # Matemática pura sin recortes
        gris = (r * 0.299) + (g * 0.587) + (b * 0.114)
        
        # Guardamos en float32 para la matemática de Fase 2
        gray_float[y, x] = gris
        # Guardamos en uint8 para exportar rápido en Fase 3
        gray_uint8[y, x] = round(min(gris, 255.0))

@cuda.jit
def sobel_cuda(gray_float, out_uint8):
    y, x = cuda.grid(2)
    # 1. Aseguramos que el hilo no se salga de la imagen
    if y < gray_float.shape[0] and x < gray_float.shape[1]:
        # 2. Procesamos el interior (dejando el borde de 1px)
        if y > 0 and y < gray_float.shape[0] - 1 and x > 0 and x < gray_float.shape[1] - 1:
            gx = 0.0
            gy = 0.0
            for ky in range(3):
                for kx in range(3):
                    p = float(gray_float[y + ky - 1, x + kx - 1])
                    gx += p * float(gx_kernel[ky][kx])
                    gy += p * float(gy_kernel[ky][kx])
            
            mag = math.sqrt(gx * gx + gy * gy)
            # Redondeo estadístico y límite
            out_uint8[y, x] = round(min(mag, 255.0))
        else:
            # 3. MARCO NEGRO: Limpiamos la "basura" de VRAM en los bordes
            out_uint8[y, x] = 0

def procesar(imagen_path, config):
    if (not os.path.exists(imagen_path)):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    try:
        imagen_ram = pil.open(imagen_path).convert("RGB")
        
        # --- WARMUP ---
        aux_matriz_rgb_cpu = np.asarray(imagen_ram, dtype=np.float32)
        aux_height, aux_width = aux_matriz_rgb_cpu.shape[:2]

        aux_rgb_gpu = cuda.to_device(aux_matriz_rgb_cpu) 
        aux_gris_f32_gpu = cuda.device_array((aux_height, aux_width), dtype=np.float32)
        aux_gris_u8_gpu = cuda.device_array((aux_height, aux_width), dtype=np.uint8)
        aux_sobel_u8_gpu = cuda.device_array((aux_height, aux_width), dtype=np.uint8)

        AUX_THREADS = (128, 8) 
        aux_blocks_y = (aux_height + AUX_THREADS[0] - 1) // AUX_THREADS[0]
        aux_blocks_x = (aux_width + AUX_THREADS[1] - 1) // AUX_THREADS[1]
        aux_blocks = (aux_blocks_y, aux_blocks_x)

        rgb_to_gray_cuda[aux_blocks, AUX_THREADS](aux_rgb_gpu, aux_gris_f32_gpu, aux_gris_u8_gpu)
        sobel_cuda[aux_blocks, AUX_THREADS](aux_gris_f32_gpu, aux_sobel_u8_gpu)
        cuda.synchronize()

        # --- REAL ---
        matriz_rgb_cpu = np.asarray(imagen_ram, dtype=np.float32)
        height, width = matriz_rgb_cpu.shape[:2] 
        
        inicio_transfer = perf_counter()
        matriz_rgb_gpu = cuda.to_device(matriz_rgb_cpu)
        cuda.synchronize() 
        t_transfer_in = perf_counter() - inicio_transfer
        
        # Pre-alocación de memoria con los tipos correctos
        matriz_gris_f32_gpu = cuda.device_array((height, width), dtype=np.float32)
        matriz_gris_u8_gpu = cuda.device_array((height, width), dtype=np.uint8)
        matriz_sobel_u8_gpu = cuda.device_array((height, width), dtype=np.uint8)
        
        # *** CAMBIA ESTO SEGÚN LA CARPETA (16_16, 128_8, etc) ***
        THREADS_PER_BLOCK = (8, 8)

        blockspergrid_y = (height + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0]
        blockspergrid_x = (width + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1]
        blockspergrid = (blockspergrid_y, blockspergrid_x)
        
    except Exception as e:
        raise ValueError(f"Falla en Fase 0: {e}")
    
    # Pasamos todas las variables de memoria pre-asignada
    datos_pipeline = (matriz_rgb_gpu, matriz_gris_f32_gpu, matriz_gris_u8_gpu, matriz_sobel_u8_gpu, height, width, blockspergrid, THREADS_PER_BLOCK)
    return datos_pipeline, t_transfer_in