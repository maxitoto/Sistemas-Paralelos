import os
from PIL import Image
import numpy as np

def procesar(matrices, config):
    # Desempaquetamos los tensores de PyTorch
    tensor_gris, tensor_gx, tensor_gy, tensor_sobel = matrices
    
    args, nombre_pipeline = config
    nombre_imagen = args.get("input").split("/")[-1].split(".")[0]

    # --- EXTRACCIÓN A NUMPY ---
    # .squeeze() elimina las dimensiones vacías del Batch y Canal (1, 1, H, W) -> (H, W)
    matrizGris = tensor_gris.squeeze().numpy()
    matrizGx = tensor_gx.squeeze().numpy()
    matrizGy = tensor_gy.squeeze().numpy()
    matrizSobel = tensor_sobel.squeeze().numpy()

    porcentaje_blancos = white_percentage(matrizSobel)
    
    address = os.path.join(nombre_pipeline, nombre_imagen)

    if not os.path.exists(address):
        os.makedirs(address)
    else:
        print(f"    [Fase 3] La carpeta '{address}/' ya estaba creada")
        return porcentaje_blancos
    
    Image.fromarray(matrizGris).save(os.path.join(address, "imagen_gris.png"))
    Image.fromarray(matrizGx).save(os.path.join(address, "gx_verticales.png"))
    Image.fromarray(matrizGy).save(os.path.join(address, "gy_horizontales.png"))
    Image.fromarray(matrizSobel).save(os.path.join(address, "sobel_magnitud_final.png"))
    
    print(f"\n [+] Las 4 imágenes fueron exportadas en la carpeta '{address}/'")

    # Retornamos únicamente el porcentaje
    return porcentaje_blancos

def white_percentage(gray: np.ndarray) -> float:
    total_pixels = gray.size
    white_pixels = np.count_nonzero(gray == 255)
    return float(white_pixels) * 100.0 / float(total_pixels)