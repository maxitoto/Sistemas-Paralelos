import numpy as np

def procesar(datos_en_transito, config):
    matriz_rgb, t_transfer_in = datos_en_transito
    try:

        r = matriz_rgb[:, :, 0]
        g = matriz_rgb[:, :, 1]
        b = matriz_rgb[:, :, 2]
        
        matrizGris = (0.299 * r) + (0.587 * g) + (0.114 * b)

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 1: {e}")
    t_transfer_in = 0
    return (matrizGris, t_transfer_in)