import math
from PIL import Image as pil
import numpy as np
import os
from numba import cuda
from .fase0 import rgb_to_gray_cuda

# fase 1 encontrar la imagen en paralelo con numba.
def procesar(datos, config):

    matriz_rgb_gpu, matriz_gris_gpu, matriz_sobel_gpu, height, width, blockspergrid, THREADS_PER_BLOCK = datos
    #calculamos solo grises

    rgb_to_gray_cuda[blockspergrid, THREADS_PER_BLOCK](matriz_rgb_gpu, matriz_gris_gpu)
    cuda.synchronize()

    return (matriz_gris_gpu, matriz_sobel_gpu, height, width, blockspergrid, THREADS_PER_BLOCK)




