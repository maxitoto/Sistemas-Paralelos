from PIL import Image as pil
import numpy as np
import os

# fase 1 encontrar la imagen en paralelo con numba.
def procesar(imagen_path, config):

    if (not os.path.exists(imagen_path)):
        raise FileExistsError(f"Error: La imagen no fue encontrada en la ruta indicada \n {imagen_path}")
    
    try:
        imagen_ram = pil.open(imagen_path)
        
        #imagen_grises = imagen_pil.convert('L') # pillow tiene una funcion para llevar la imagen a escala de grises (no lo voy a usar)
        
        # Llevamos la imagen a una matriz 3D (Alto x Ancho x 3 (Red,Green,Blue)) en float32 para cálculos
        matriz_rgb = np.asarray(imagen_ram, dtype=np.float32)

        # Vectorización, se multiplica cada color de cada pixel en cada fila de al mismo "tiempo".
        # se suman para obterner la matriz completa y gris.
        matrizGris = (0.299 * matriz_rgb[:, :, 0]) + (0.587 * matriz_rgb[:, :, 1]) + (0.114 * matriz_rgb[:, :, 2])

        # Aseguramos que los valores estén en el rango 0-255 y pasamos a enteros de 8 bits
        matrizGris = np.clip(matrizGris, 0, 255).astype(np.uint8)


    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 1: No se pudo procesar el archivo. Error: {e}")
    
    return matrizGris
