# Importamos el kernel precompilado
from .fase0 import calcular_gris

def procesar(datos_en_transito, config):
    try:
        datos, t_transfer_in = datos_en_transito
        matriz_rgb, alto, ancho = datos
        
        # Ejecutamos la función ya compilada en C++ por Numba
        matrizGris = calcular_gris(matriz_rgb, alto, ancho)

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 1: {e}")
    
    return (matrizGris, alto, ancho)