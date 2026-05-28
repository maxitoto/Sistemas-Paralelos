# Importamos el kernel precompilado
from .fase0 import kernel_sobel

def procesar(datos, config):
    try:
        matrizGris, alto, ancho = datos
        
        # Ejecución JIT nativa multi-hilo
        matrizGx, matrizGy, matrizSobel = kernel_sobel(matrizGris, alto, ancho)

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 2: {e}")

    # Devolvemos las 4 matrices para la Fase 3
    return (matrizGris, matrizGx, matrizGy, matrizSobel)