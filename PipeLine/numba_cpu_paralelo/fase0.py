import os
from PIL import Image as pil
import numpy as np
from numba import njit, prange
import math
from time import perf_counter

# =================================================================
# --- DEFINICIÓN DE KERNELS NUMBA (Se compilan y exportan desde aquí) ---
# =================================================================

@njit(parallel=True)
def calcular_gris(matriz_rgb, alto, ancho):
    matrizGris = np.zeros((alto, ancho), dtype=np.uint8)
    for fil in prange(alto):
        for col in range(ancho):
            r = matriz_rgb[fil, col, 0]
            g = matriz_rgb[fil, col, 1]
            b = matriz_rgb[fil, col, 2]
            gris = (r * 0.299) + (g * 0.587) + (b * 0.114)
            matrizGris[fil, col] = int(gris)
    return matrizGris

@njit(parallel=True)
def kernel_sobel(matrizGris, alto, ancho):
    matrizGx = np.zeros((alto, ancho), dtype=np.uint8)
    matrizGy = np.zeros((alto, ancho), dtype=np.uint8)
    matrizSobel = np.zeros((alto, ancho), dtype=np.uint8)

    for fil in prange(1, alto-1):
        for col in range(1, ancho-1):
            p00 = float(matrizGris[fil-1, col-1])
            p01 = float(matrizGris[fil-1, col])
            p02 = float(matrizGris[fil-1, col+1])
            
            p10 = float(matrizGris[fil, col-1])
            p12 = float(matrizGris[fil, col+1])
            
            p20 = float(matrizGris[fil+1, col-1])
            p21 = float(matrizGris[fil+1, col])
            p22 = float(matrizGris[fil+1, col+1])

            KerGx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20)
            KerGy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02)

            matrizGx[fil, col] = int(min(abs(KerGx), 255.0))
            matrizGy[fil, col] = int(min(abs(KerGy), 255.0))
            matrizSobel[fil, col] = int(min(math.sqrt(KerGx**2 + KerGy**2), 255.0))
            
    return matrizGx, matrizGy, matrizSobel

# =================================================================
# --- FASE 0: PREPARACIÓN DE DATOS Y CALENTAMIENTO ---
# =================================================================

def procesar(imagen_path, config):
    if not os.path.exists(imagen_path):
        raise FileExistsError(f"Error: La imagen no fue encontrada \n {imagen_path}")
    
    try:
        # 1. Carga de imagen en RGB
        imagen_ram = pil.open(imagen_path).convert("RGB")
        matriz_rgb = np.asarray(imagen_ram, dtype=np.uint8)
        
        alto = matriz_rgb.shape[0]
        ancho = matriz_rgb.shape[1]

        # 2. WARMUP (CALENTAMIENTO Y COMPILACIÓN JIT)
        # Forzamos a Numba a compilar el código de C++ pasándole los tipos de datos reales.
        # Esto toma el impacto de compilación aquí, fuera del cronómetro oficial.
        _aux_gris = calcular_gris(matriz_rgb, alto, ancho)
        _aux_gx, _aux_gy, _aux_sobel = kernel_sobel(_aux_gris, alto, ancho)

        t_transfer_in = 0

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 0: {e}")
    
    datos = (matriz_rgb, alto, ancho)
    return (datos, t_transfer_in)