from .fase0 import calcular_gris

def procesar(datos, config):
    matriz_rgb, alto, ancho = datos
    return calcular_gris(matriz_rgb, alto, ancho), alto, ancho