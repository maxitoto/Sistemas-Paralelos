# Importamos el kernel precompilado
from .fase0 import kernel_sobel
import numpy as np

def procesar(datos, config):
    try:
        matrizGris, alto, ancho = datos
        
        # Ejecución JIT nativa multi-hilo (Devuelve uint8 gracias a la corrección en Fase 0)
        matrizGx, matrizGy, matrizSobel = kernel_sobel(matrizGris, alto, ancho)

        # Casteo final de la matriz gris a uint8 para la exportación en Fase 3,
        # usando el mismo redondeo estadístico de Numpy para no perder fidelidad.
        mGris_uint8 = np.clip(np.round(matrizGris), 0, 255).astype(np.uint8)

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 2: {e}")

    # Devolvemos las 4 matrices ultraligeras para la Fase 3
    return (mGris_uint8, matrizGx, matrizGy, matrizSobel)