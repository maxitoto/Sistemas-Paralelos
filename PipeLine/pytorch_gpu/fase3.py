import os
from PIL import Image
import numpy as np
import torch
from time import perf_counter # IMPORTAMOS EL CRONÓMETRO

def procesar(matrices, config):
    try:
        d_img_gris, d_img_sobel = matrices
        
        # --- MEDICIÓN DE TRANSFERENCIA (GPU -> CPU) ---
        inicio_transfer = perf_counter()
        
        # El equivalente a matriz_gris_gpu.copy_to_host() en PyTorch
        matriz_gris_cpu = d_img_gris.squeeze().to(torch.uint8).cpu().numpy()
        matriz_sobrel_cpu = d_img_sobel.squeeze().to(torch.uint8).cpu().numpy()
        
        t_transfer_out = perf_counter() - inicio_transfer

        args, nombre_pipeline = config
        nombre_imagen = args.get("input").split("/")[-1].split(".")[0]

        # Usamos directamente el resultado de la función para evitar el doble cálculo
        porcentaje_blancos = white_percentage(matriz_sobrel_cpu)
        
        address = os.path.join(nombre_pipeline, nombre_imagen)

        if not os.path.exists(address):
            os.makedirs(address)
        else:
            print(f"    [Fase 3] La carpeta '{address}/' ya estaba creada")
            return porcentaje_blancos, t_transfer_out # Devolvemos tiempo también aquí
        
        Image.fromarray(matriz_gris_cpu).save(os.path.join(address, "imagen_gris.png"))
        Image.fromarray(matriz_sobrel_cpu).save(os.path.join(address, "imagen_sobel.png"))
        
        print(f"\n [+] Las 2 imágenes fueron exportadas en la carpeta '{address}/'")

        # IMPORTANTE: Devolvemos una tupla con el tiempo
        return porcentaje_blancos, t_transfer_out

    except Exception as e:
        raise RuntimeError(f"Falla Crítica en Fase 3: {e}")

def white_percentage(gray: np.ndarray) -> float:
    total_pixels = gray.size
    white_pixels = np.count_nonzero(gray == 255)
    return float(white_pixels) * 100.0 / float(total_pixels)