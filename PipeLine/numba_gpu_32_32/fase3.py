import os
from PIL import Image
import numpy as np
from time import perf_counter

def procesar(matrices, config):
    matriz_gris_u8_gpu, matriz_sobel_u8_gpu = matrices
    
    # --- MEDICIÓN DE TRANSFERENCIA (GPU -> CPU) ---
    inicio_transfer = perf_counter()
    
    # Esta copia ahora pesa 1/4 del original y viaja más rápido
    matriz_gris_cpu = matriz_gris_u8_gpu.copy_to_host()
    matriz_sobel_cpu = matriz_sobel_u8_gpu.copy_to_host()
    
    t_transfer_out = perf_counter() - inicio_transfer

    args, nombre_pipeline = config
    nombre_imagen = args.get("input").split("/")[-1].split(".")[0]

    # Doble división FIXEADA
    porcentaje_blancos = white_percentage(matriz_sobel_cpu)
    
    address = os.path.join(nombre_pipeline, nombre_imagen)

    if not os.path.exists(address):
        os.makedirs(address)
    else:
        print(f"    [Fase 3] La carpeta '{address}/' ya estaba creada")
        return porcentaje_blancos, t_transfer_out 
    
    Image.fromarray(matriz_gris_cpu).save(os.path.join(address, "imagen_gris.png"))
    Image.fromarray(matriz_sobel_cpu).save(os.path.join(address, "imagen_sobel.png"))
    
    # print(f"\n [+] Las 2 imágenes fueron exportadas en la carpeta '{address}/'")

    return porcentaje_blancos, t_transfer_out

def white_percentage(gray: np.ndarray) -> float:
    total_pixels = gray.size
    white_pixels = np.count_nonzero(gray == 255)
    return float(white_pixels) * 100.0 / float(total_pixels)