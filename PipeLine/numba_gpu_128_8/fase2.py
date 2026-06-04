from numba import cuda
from .fase0 import sobel_cuda

def procesar(datos, config):
    matriz_gris_f32_gpu, matriz_gris_u8_gpu, matriz_sobel_u8_gpu, height, width, blockspergrid, THREADS_PER_BLOCK = datos

    # Leemos de F32 (alta precisión) y guardamos en U8 (memoria ultraligera)
    sobel_cuda[blockspergrid, THREADS_PER_BLOCK](matriz_gris_f32_gpu, matriz_sobel_u8_gpu)
    cuda.synchronize()

    # Solo enviamos a la Fase 3 las matrices aligeradas a 1 byte
    return (matriz_gris_u8_gpu, matriz_sobel_u8_gpu)