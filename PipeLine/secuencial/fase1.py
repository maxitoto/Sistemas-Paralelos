import numpy as np

def procesar(datos, config):
    matriz_rgb, alto, ancho = datos
    matrizGris = np.zeros((alto, ancho), dtype=np.uint8)
    
    # Conversión secuencial optimizada
    for fil in range(alto):
        for col in range(ancho):
            matrizGris[fil, col] = int(
                matriz_rgb[fil, col, 0] * 0.299 + 
                matriz_rgb[fil, col, 1] * 0.587 + 
                matriz_rgb[fil, col, 2] * 0.114
            )
    return (matrizGris, alto, ancho)