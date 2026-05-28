import os
from PIL import Image
import numpy as np
from time import perf_counter

def procesar(matrices, config):
    try:
        matrizGris, matrizGx, matrizGy, matrizSobel = matrices

        t_transfer_out = 0

        args, nombre_pipeline = config
        nombre_imagen = args.get("input").split("/")[-1].split(".")[0]

        # Calculamos % de Pixeles blancos
        porcentaje_blancos = white_percentage(matrizSobel)
        
        address = os.path.join(nombre_pipeline, nombre_imagen)

        if not os.path.exists(address):
            os.makedirs(address)
        else:
            print(f"    [Fase 3] La carpeta '{address}/' ya estaba creada")
            return porcentaje_blancos, t_transfer_out
        
        Image.fromarray(matrizGris).save(os.path.join(address, "imagen_gris.png"))
        Image.fromarray(matrizGx).save(os.path.join(address, "gx_verticales.png"))
        Image.fromarray(matrizGy).save(os.path.join(address, "gy_horizontales.png"))
        Image.fromarray(matrizSobel).save(os.path.join(address, "sobel_magnitud_final.png"))
        
        # print(f"\n [+] Las 4 imágenes fueron exportadas en la carpeta '{address}/'")

        return porcentaje_blancos, t_transfer_out

    except Exception as e:
        raise RuntimeError(f"Falla Crítica en Fase 3: {e}")

def white_percentage(gray: np.ndarray) -> float:
    total_pixels = gray.size
    white_pixels = np.count_nonzero(gray == 255)
    return float(white_pixels) * 100.0 / float(total_pixels)