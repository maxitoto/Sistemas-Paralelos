import numpy as np

def procesar(datos, config):
    # El orquestador ya le quitó el 0.0, recibimos directamente los datos
    matriz_rgb, alto, ancho = datos

    # Nace como float32 para la matemática precisa
    matrizGris = np.zeros((alto, ancho), dtype=np.float32)
    
    # Conversión puramente secuencial (con bucles FOR)
    for fil in range(alto):
        for col in range(ancho):
            matrizGris[fil][col] = (
                (matriz_rgb[fil, col, 0] * 0.299) + 
                (matriz_rgb[fil, col, 1] * 0.587) + 
                (matriz_rgb[fil, col, 2] * 0.114)
            )
            
    return (matrizGris, alto, ancho)