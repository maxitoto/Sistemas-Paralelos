import os
from PIL import Image
import numpy as np
import torch
from time import perf_counter

def procesar(matrices, config):
    try:
        # Desempaquetamos las 4 matrices que vienen de Fase 2
        d_img_gris, d_img_gx, d_img_gy, d_img_sobel = matrices
        
        # --- MEDICIÓN DE TRANSFERENCIA (GPU -> CPU) ---
        inicio_transfer = perf_counter()
        
        # Ya vienen en uint8 desde Fase 2, así que solo extraemos a la RAM
        # .squeeze() elimina dimensiones sobrantes (Batch y Canales) -> (Alto, Ancho)
        matriz_gris_cpu = d_img_gris.squeeze().cpu().numpy()
        matriz_gx_cpu = d_img_gx.squeeze().cpu().numpy()
        matriz_gy_cpu = d_img_gy.squeeze().cpu().numpy()
        matriz_sobrel_cpu = d_img_sobel.squeeze().cpu().numpy()
        
        t_transfer_out = perf_counter() - inicio_transfer

        args, nombre_pipeline = config
        nombre_imagen = args.get("input").split("/")[-1].split(".")[0]

        # Calculamos %
        porcentaje_blancos = white_percentage(matriz_sobrel_cpu)
        
        address = os.path.join(nombre_pipeline, nombre_imagen)

        if not os.path.exists(address):
            os.makedirs(address)
        else:
            print(f"    [Fase 3] La carpeta '{address}/' ya estaba creada")
            return porcentaje_blancos, t_transfer_out
        
        Image.fromarray(matriz_gris_cpu).save(os.path.join(address, "imagen_gris.png"))
        Image.fromarray(matriz_gx_cpu).save(os.path.join(address, "gx_verticales.png"))
        Image.fromarray(matriz_gy_cpu).save(os.path.join(address, "gy_horizontales.png"))
        Image.fromarray(matriz_sobrel_cpu).save(os.path.join(address, "sobel_magnitud_final.png"))
        
        print(f"\n [+] Las 4 imágenes fueron exportadas en la carpeta '{address}/'")

        return porcentaje_blancos, t_transfer_out

    except Exception as e:
        raise RuntimeError(f"Falla Crítica en Fase 3: {e}")

def white_percentage(gray: np.ndarray) -> float:
    total_pixels = gray.size
    white_pixels = np.count_nonzero(gray == 255)
    return float(white_pixels) * 100.0 / float(total_pixels)