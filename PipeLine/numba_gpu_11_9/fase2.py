import numpy as np
from numba import cuda
import math
from .fase0 import sobel_cuda

gx_kernel = (
    (-1, 0, 1),
    (-2, 0, 2),
    (-1, 0, 1),
)

gy_kernel = (
    (1, 2, 1),
    (0, 0, 0),
    (-1, -2, -1),
)

def procesar(datos, config):

    matriz_gris_gpu, matriz_sobel_gpu, height, width, blockspergrid, THREADS_PER_BLOCK = datos

    #calculamos solo sobel
    sobel_cuda[blockspergrid, THREADS_PER_BLOCK](matriz_gris_gpu, matriz_sobel_gpu)
    cuda.synchronize()

    return (matriz_gris_gpu, matriz_sobel_gpu)
