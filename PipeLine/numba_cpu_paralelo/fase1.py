# Importamos el kernel precompilado
from .fase0 import calcular_gris

def procesar(datos, config):
    try:
        matriz_rgb, alto, ancho = datos
        
        # Ejecutamos la función ya compilada en C++ por Numba
        matrizGris = calcular_gris(matriz_rgb, alto, ancho)

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 1: {e}")
    
    return (matrizGris, alto, ancho)