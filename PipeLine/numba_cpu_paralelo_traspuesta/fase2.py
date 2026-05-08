import numpy as np
import math
from numba import njit, prange

def procesar(datos_entrada, config):
    matrizGris, alto, ancho = datos_entrada
    
    # 1. Calculamos Gx (Bordes Verticales) pasando la matriz normal
    mGx_float = calcular_gx_numba(matrizGris)
    
    # 2. Calculamos Gy (Bordes Horizontales). 
    # Usamos la magia matemática: Gy es igual a aplicar Gx sobre la imagen traspuesta, 
    # y luego trasponer el resultado para enderezarlo. ¡Cero costo de memoria!
    mGy_float = calcular_gx_numba(matrizGris.T).T
    
    # 3. Calculamos la magnitud y normalizamos a 8 bits
    mGris, mGx, mGy, mSobel = calcular_magnitud(matrizGris, mGx_float, mGy_float)
    
    return (mGris, mGx, mGy, mSobel)

@njit(parallel=True)
def calcular_gx_numba(matriz):
    # Ojo: Si la matriz entra traspuesta, 'alto' y 'ancho' se invierten automáticamente aquí.
    alto, ancho = matriz.shape
    gx = np.zeros((alto, ancho), dtype=np.float32)
    
    # Aplicamos el kernel de derivada en X estándar
    for f in prange(1, alto-1):
        for c in range(1, ancho-1):
            gx[f, c] = (matriz[f-1, c+1] + 2.0 * matriz[f, c+1] + matriz[f+1, c+1]) - \
                       (matriz[f-1, c-1] + 2.0 * matriz[f, c-1] + matriz[f+1, c-1])
                       
    return gx

@njit(parallel=True)
def calcular_magnitud(matrizGris, gx_float, gy_float):
    alto, ancho = matrizGris.shape
    
    mGx = np.zeros((alto, ancho), dtype=np.uint8)
    mGy = np.zeros((alto, ancho), dtype=np.uint8)
    mSobel = np.zeros((alto, ancho), dtype=np.uint8)
    
    for f in prange(alto):
        for c in range(ancho):
            x = gx_float[f, c]
            y = gy_float[f, c]
            
            # Normalizamos
            mGx[f, c] = int(min(abs(x), 255.0))
            mGy[f, c] = int(min(abs(y), 255.0))
            
            # Magnitud
            mag = math.sqrt(x**2 + y**2)
            mSobel[f, c] = int(min(mag, 255.0))
            
    return matrizGris, mGx, mGy, mSobel