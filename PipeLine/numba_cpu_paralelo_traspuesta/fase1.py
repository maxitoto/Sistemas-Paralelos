from PIL import Image as pil
import numpy as np
import os
from numba import njit, prange

# fase 1 encontrar la imagen en paralelo con numba.
def procesar(imagen_path, config):

    if (not os.path.exists(imagen_path)):
        raise FileExistsError(f"Error: La imagen no fue encontrada en la ruta indicada \n {imagen_path}")
    
    try:
        imagen_ram = pil.open(imagen_path)
        
        #imagen_grises = imagen_pil.convert('L') # pillow tiene una funcion para llevar la imagen a escala de grises (no lo voy a usar)
        
        matriz_rgb = np.array(imagen_ram) # usando numpy llevamos la imgen a una matriz donde es 3(Red Green Blue) x Ancho x Alto

        #alto y ancho de la imagen
        alto = len(matriz_rgb)
        ancho = len(matriz_rgb[0])

        matrizGris = calcular(matriz_rgb, alto, ancho)

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 1: No se pudo procesar el archivo. Error: {e}")
    
    return (matrizGris, alto, ancho)

@njit(parallel = True)
def calcular(matriz_rgb, alto, ancho):
    matrizGris = np.zeros((alto, ancho), dtype=np.uint8) # matriz de ceros para hacer el gris (mismo tamaño que la imagen)

    #recorrer cada pixel y hacerlo gris
    for fil in prange(alto):
        for col in range(ancho):
            r = matriz_rgb[fil][col][0]
            g = matriz_rgb[fil][col][1]
            b =matriz_rgb[fil][col][2]

            gris = (r * 0.299) + (g * 0.587) + (b * 0.114)

            matrizGris[fil][col] = int(gris)

    return matrizGris