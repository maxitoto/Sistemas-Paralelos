import numpy as np
from numba import njit, prange
from .fase0 import calcular_gx, combinar


def procesar(datos, config):
    matrizGris, alto, ancho = datos
    
    # 1. Gx normal
    gx = calcular_gx(matrizGris.astype(np.float32))
    # 2. Gy usando traspuesta (el secreto de eficiencia)
    gy = calcular_gx(matrizGris.astype(np.float32).T).T
    
    return combinar(matrizGris, gx, gy)