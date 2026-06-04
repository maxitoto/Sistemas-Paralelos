import os
from PIL import Image
import numpy as np

def procesar(matrices, config):
    tensor_gris, tensor_gx, tensor_gy, tensor_sobel = matrices
    
    args, nombre_pipeline = config
    nombre_imagen = args.get("input").split("/")[-1].split(".")[0]

    # Ya están en uint8
    matrizGris = tensor_gris.squeeze().numpy()
    matrizGx = tensor_gx.squeeze().numpy()
    matrizGy = tensor_gy.squeeze().numpy()
    matrizSobel = tensor_sobel.squeeze().numpy()

    porcentaje_blancos = white_percentage(matrizSobel)
    
    address = os.path.join(nombre_pipeline, nombre_imagen)
    os.makedirs(address, exist_ok=True)
    
    Image.fromarray(matrizGris).save(os.path.join(address, "imagen_gris.png"))
    Image.fromarray(matrizGx).save(os.path.join(address, "gx_verticales.png"))
    Image.fromarray(matrizGy).save(os.path.join(address, "gy_horizontales.png"))
    Image.fromarray(matrizSobel).save(os.path.join(address, "sobel_magnitud_final.png"))
    
    t_transfer_out = 0.0
    return porcentaje_blancos, t_transfer_out

def white_percentage(gray: np.ndarray) -> float:
    total_pixels = gray.size
    white_pixels = np.count_nonzero(gray == 255)
    return float(white_pixels) * 100.0 / float(total_pixels)