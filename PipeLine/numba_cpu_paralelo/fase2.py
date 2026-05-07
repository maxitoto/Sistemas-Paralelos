import numpy as np
import math
from numba import njit, prange

def procesar(datos_entrada, config):

    return  kernel_sobel(datos_entrada)


@njit(parallel = True)
def kernel_sobel(matrizGris):
    
    matrizGris, alto, ancho = matrizGris  # vuelvo a obtener el alto y ancho (repetirme, pero + legible)
    
    matrizGx = np.zeros((alto, ancho), dtype=np.uint8)
    matrizGy = np.zeros((alto, ancho), dtype=np.uint8)
    matrizSobel = np.zeros((alto, ancho), dtype=np.uint8)

    #ahora defino los Kernels (ventanas para filtro sobel)

    #Gx = np.array([ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ], dtype=np.float32)

    #Gy = np.array([ [-1, -2, -1], [0, 0, 0], [1, 2, 1] ], dtype=np.float32)

    # debemos ignorar los bordes extremos de la imagen gris, porque produce desbordamineto al queres mover el kerner por toda la imagen
    for fil in prange(1, alto-1):
        for col in range(1, ancho-1):
            
            #obtengo los pixeles vecinos, son 8 pixeles.
            p00 = float(matrizGris[fil-1, col-1])
            p01 = float(matrizGris[fil-1, col])
            p02 = float(matrizGris[fil-1, col+1])
            
            p10 = float(matrizGris[fil, col-1])
            p12 = float(matrizGris[fil, col+1])
            
            p20 = float(matrizGris[fil+1, col-1])
            p21 = float(matrizGris[fil+1, col])
            p22 = float(matrizGris[fil+1, col+1])

            
            KerGx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20)#obtengo el pixel multiplicado por el kernel x
            KerGy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02)#obtengo el pixel multiplicado por el kernel y

            matrizGx[fil, col] = int(min(abs(KerGx), 255.0))
            matrizGy[fil, col] = int(min(abs(KerGy), 255.0))
            matrizSobel[fil, col] = int(min(math.sqrt(KerGx**2 + KerGy**2), 255.0)) #paso 4, pero mejor lo incorporo aquí
                
    return (matrizGris, matrizGx, matrizGy, matrizSobel)