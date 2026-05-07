#imagenes resultado de Gx, Gy y Gx+Gy

import os
from PIL import Image

def procesar(matrices, config):
    
    matrizGris, matrizGx, matrizGy, matrizSobel = matrices

    args, nombre_pipeline = config

    # Obtenemos nombres desde la configuración del orquestador
    nombre_imagen = args.get("input").split("/")[-1].split(".")[0]

    # Calculamos % de Pixeles blancos
    cantidad_blancos = sum(pixel == 255 for fila in matrizSobel for pixel in fila)
    alto, ancho = matrizGris.shape
    total_pixeles = alto * ancho
    porcentaje_blancos = (cantidad_blancos / total_pixeles) * 100.0
    
    # Estructura: nombre_del_pipeline/nombre_de_la_imagen/
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
    
    print(f"\n [+] Las 3 imágenes fueron exportadas en la carpeta '{address}/'")

    return porcentaje_blancos