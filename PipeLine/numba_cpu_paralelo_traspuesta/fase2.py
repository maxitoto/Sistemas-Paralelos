import numpy as np
import math
from numba import njit, prange

def procesar(datos_entrada, config):
    # Desempaquetamos la tupla (matrizGris, alto, ancho) que viene de Fase 1
    matrizGris, alto, ancho = datos_entrada
    
    # Ejecutamos la lógica de traspuesta paralela
    mGris, mGx, mGy, mSobel = kernel_sobel_traspuesta(matrizGris, alto, ancho)
    
    return (mGris, mGx, mGy, mSobel)

@njit(parallel = True)
def aplicar_1d_y_trasponer(matriz, h, w, k0, k1, k2):
    # Creamos la matriz de destino YA traspuesta (w x h)
    # Esto asegura que la siguiente pasada también sea horizontal (óptimo para caché)
    res_t = np.zeros((w, h), dtype=np.float32)
    
    for f in prange(h):
        for c in range(1, w - 1):
            # Operación 1D horizontal: k0, k1, k2 son los valores del kernel separable
            res_t[c, f] = matriz[f, c-1] * k0 + matriz[f, c] * k1 + matriz[f, c+1] * k2
            
    return res_t

@njit(parallel = True)
def kernel_sobel_traspuesta(matrizGris, alto, ancho):
    # Definimos los kernels separables 1D
    # Para Sobel: uno suaviza [1, 2, 1] y el otro deriva [-1, 0, 1]
    k_der = [-1.0, 0.0, 1.0]
    k_sua = [1.0, 2.0, 1.0]

    # --- CÁLCULO Gx (Bordes Verticales) ---
    # 1. Pasada horizontal con derivada -> Traspone
    gx_h_t = aplicar_1d_y_trasponer(matrizGris.astype(np.float32), alto, ancho, k_der[0], k_der[1], k_der[2])
    # 2. Pasada horizontal (que era la vertical original) con suavizado -> Traspone (vuelve al eje original)
    gx_v = aplicar_1d_y_trasponer(gx_h_t, ancho, alto, k_sua[0], k_sua[1], k_sua[2])

    # --- CÁLCULO Gy (Bordes Horizontales) ---
    # 1. Pasada horizontal con suavizado -> Traspone
    gy_h_t = aplicar_1d_y_trasponer(matrizGris.astype(np.float32), alto, ancho, k_sua[0], k_sua[1], k_sua[2])
    # 2. Pasada horizontal con derivada -> Traspone (vuelve al eje original)
    gy_v = aplicar_1d_y_trasponer(gy_h_t, ancho, alto, k_der[0], k_der[1], k_der[2])

    # --- MAGNITUD Y NORMALIZACIÓN ---
    mGx = np.zeros((alto, ancho), dtype=np.uint8)
    mGy = np.zeros((alto, ancho), dtype=np.uint8)
    mSobel = np.zeros((alto, ancho), dtype=np.uint8)

    for f in prange(alto):
        for c in range(ancho):
            val_x = gx_v[f, c]
            val_y = gy_v[f, c]
            
            mGx[f, c] = int(min(abs(val_x), 255.0))
            mGy[f, c] = int(min(abs(val_y), 255.0))
            
            mag = math.sqrt(val_x**2 + val_y**2)
            mSobel[f, c] = int(min(mag, 255.0))

    return (matrizGris, mGx, mGy, mSobel)