import os
from PIL import Image
import numpy as np
import torch
from time import perf_counter

def procesar(matrices, config):
    tensor_gris_gpu, tensor_sobel_gpu = matrices
    
    # medición de transferencia (GPU -> CPU)
    inicio_transfer = perf_counter()
    
    matriz_gris_cpu = tensor_gris_gpu.squeeze().to(torch.uint8).cpu().numpy()
    matriz_sobel_cpu = tensor_sobel_gpu.squeeze().to(torch.uint8).cpu().numpy()
    
    t_transfer_out = perf_counter() - inicio_transfer

    args, nombre_pipeline = config
    nombre_imagen = args.get("input", "imagen").split("/")[-1].split(".")[0]

    cantidad_blancos = np.count_nonzero(matriz_sobel_cpu == 255)
    total_pixeles = matriz_sobel_cpu.size
    porcentaje_blancos = (float(cantidad_blancos) / total_pixeles) * 100.0
    
    address = os.path.join(nombre_pipeline, nombre_imagen)
    os.makedirs(address, exist_ok=True) 
    
    Image.fromarray(matriz_gris_cpu).save(os.path.join(address, "imagen_gris.png"))
    Image.fromarray(matriz_sobel_cpu).save(os.path.join(address, "imagen_sobel.png"))
    
    print(f"\n [+] Las 2 imágenes fueron exportadas en la carpeta '{address}/'")

    return porcentaje_blancos, t_transfer_out