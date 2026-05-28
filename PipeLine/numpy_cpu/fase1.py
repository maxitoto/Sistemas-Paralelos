import numpy as np

def procesar(matriz_rgb, config):
    try:
        # Vectorización: se extraen canales y se multiplican simultáneamente
        r = matriz_rgb[:, :, 0]
        g = matriz_rgb[:, :, 1]
        b = matriz_rgb[:, :, 2]
        
        matrizGris = (0.299 * r) + (0.587 * g) + (0.114 * b)

        # Aseguramos el rango y pasamos a uint8 (exactamente tu lógica original)
        matrizGris = np.clip(matrizGris, 0, 255).astype(np.uint8)

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 1: {e}")
    
    return matrizGris