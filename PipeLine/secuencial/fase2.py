import numpy as np
import math

def procesar(datos, config):
    matrizGris, alto, ancho = datos
    matrizGx = np.zeros((alto, ancho), dtype=np.float32)
    matrizGy = np.zeros((alto, ancho), dtype=np.float32)
    matrizSobel = np.zeros((alto, ancho), dtype=np.uint8)

    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    for fil in range(1, alto - 1):
        for col in range(1, ancho - 1):
            ker_gx = 0
            ker_gy = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    pixel = float(matrizGris[fil + ky, col + kx])
                    ker_gx += Gx[ky + 1][kx + 1] * pixel
                    ker_gy += Gy[ky + 1][kx + 1] * pixel
            
            matrizGx[fil, col] = abs(ker_gx)
            matrizGy[fil, col] = abs(ker_gy)
            matrizSobel[fil, col] = int(min(math.sqrt(ker_gx**2 + ker_gy**2), 255))
            
    return (matrizGris, matrizGx, matrizGy, matrizSobel)