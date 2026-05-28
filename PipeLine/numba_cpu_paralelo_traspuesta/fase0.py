import os
from PIL import Image as pil
import numpy as np
from numba import njit, prange
from time import perf_counter

@njit(parallel=True)
def calcular_gris(matriz_rgb, alto, ancho):
    matrizGris = np.zeros((alto, ancho), dtype=np.uint8)
    for fil in prange(alto):
        for col in range(ancho):
            gris = (matriz_rgb[fil, col, 0] * 0.299) + \
                   (matriz_rgb[fil, col, 1] * 0.587) + \
                   (matriz_rgb[fil, col, 2] * 0.114)
            matrizGris[fil, col] = int(gris)
    return matrizGris

@njit(parallel=True)
def calcular_gx(matriz):
    alto, ancho = matriz.shape
    gx = np.zeros((alto, ancho), dtype=np.float32)
    for f in prange(1, alto-1):
        for c in range(1, ancho-1):
            gx[f, c] = (matriz[f-1, c+1] + 2.0 * matriz[f, c+1] + matriz[f+1, c+1]) - \
                       (matriz[f-1, c-1] + 2.0 * matriz[f, c-1] + matriz[f+1, c-1])
    return gx

@njit(parallel=True)
def combinar(matrizGris, gx, gy):
    alto, ancho = matrizGris.shape
    sobel = np.zeros((alto, ancho), dtype=np.uint8)
    mGx = np.abs(gx).astype(np.uint8)
    mGy = np.abs(gy).astype(np.uint8)
    for f in prange(alto):
        for c in range(ancho):
            sobel[f, c] = int(min(np.sqrt(gx[f, c]**2 + gy[f, c]**2), 255.0))
    return matrizGris, mGx, mGy, sobel

def procesar(imagen_path, config):
    if not os.path.exists(imagen_path):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    imagen_ram = pil.open(imagen_path).convert("RGB")
    matriz_rgb = np.asarray(imagen_ram, dtype=np.uint8)
    alto, ancho = matriz_rgb.shape[:2]

    # Warmup
    _g = calcular_gris(matriz_rgb, alto, ancho)
    _ = calcular_gx(_g)

    t_transfer_in = 0
    
    return (matriz_rgb, alto, ancho), t_transfer_in