# imagenes resultado de Gx, Gy y Gx+Gy - Versión por Experimento
import os
from PIL import Image

def procesar(matrices, config):
    matrizGris, matrizGx, matrizGy, matrizSobel = matrices

    # Extraemos dimensiones directamente de la lista
    alto = len(matrizGris)
    ancho = len(matrizGris[0])

    args, nombre_pipeline = config

    # Obtenemos nombres desde la configuración del orquestador
    nombre_imagen = args.get("input").split("/")[-1].split(".")[0]
    
    cantidad_blancos = sum(pixel == 255 for fila in matrizSobel for pixel in fila)
    total_pixeles = alto * ancho
    porcentaje_blancos = (cantidad_blancos / total_pixeles) * 100.0

    # Estructura: nombre_del_pipeline/nombre_de_la_imagen/
    address = os.path.join(nombre_pipeline, nombre_imagen)

    if not os.path.exists(address):
        os.makedirs(address)
    else:
        print(f"    [Fase 3] La carpeta '{address}/' ya estaba creada")
        return porcentaje_blancos

    
    def guardar_imagen(matriz_2d, nombre_archivo):
        # Aplanamos la matriz para Pillow (100% Python puro)
        datos_planos = [pixel for fila in matriz_2d for pixel in fila]
        img = Image.new('L', (ancho, alto))
        img.putdata(datos_planos)
        img.save(os.path.join(address, nombre_archivo))

    # Guardamos el set completo para este experimento
    guardar_imagen(matrizGris, "1_gris.png")
    guardar_imagen(matrizGx, "2_gx_verticales.png")
    guardar_imagen(matrizGy, "3_gy_horizontales.png")
    guardar_imagen(matrizSobel, "4_sobel_final.png")
    
    print(f"    [Fase 3] Imágenes guardadas en: {address}")

    return porcentaje_blancos