#imagenes resultado de Gx, Gy y Gx+Gy

import os
from PIL import Image
import numpy as np

def procesar(matrices, config):
    
    matriz_gris_gpu, matriz_sobel_gpu = matrices
    matriz_gris_cpu = matriz_gris_gpu.copy_to_host()
    matriz_sobrel_cpu = matriz_sobel_gpu.copy_to_host()

    args, nombre_pipeline = config

    # Obtenemos nombres desde la configuración del orquestador
    nombre_imagen = args.get("input").split("/")[-1].split(".")[0]

    # Calculamos % de Pixeles blancos
    cantidad_blancos = white_percentage(matriz_sobrel_cpu)
    alto, ancho = matriz_sobrel_cpu.shape
    total_pixeles = alto * ancho
    porcentaje_blancos = (cantidad_blancos / total_pixeles) * 100.0
    
    # Estructura: nombre_del_pipeline/nombre_de_la_imagen/
    address = os.path.join(nombre_pipeline, nombre_imagen)

    if not os.path.exists(address):
        os.makedirs(address)
    else:
        print(f"    [Fase 3] La carpeta '{address}/' ya estaba creada")
        return porcentaje_blancos
    
    Image.fromarray(matriz_gris_cpu).save(os.path.join(address, "imagen_gris.png"))
    Image.fromarray(matriz_sobrel_cpu).save(os.path.join(address, "imagen_sobel.png"))
    
    print(f"\n [+] Las 2 imágenes fueron exportadas en la carpeta '{address}/'")

    return porcentaje_blancos

def white_percentage(gray: np.ndarray) -> float:
    total_pixels = gray.size
    white_pixels = np.count_nonzero(gray == 255)
    return float(white_pixels) * 100.0 / float(total_pixels)