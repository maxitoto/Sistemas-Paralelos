from numba import cuda
from .fase0 import rgb_to_gray_cuda

def procesar(datos, config):
    matriz_rgb_gpu, matriz_gris_f32_gpu, matriz_gris_u8_gpu, matriz_sobel_u8_gpu, height, width, blockspergrid, THREADS_PER_BLOCK = datos
    
    # La GPU escribe el float para la Fase 2 y el uint8 para la Fase 3 simultáneamente
    rgb_to_gray_cuda[blockspergrid, THREADS_PER_BLOCK](matriz_rgb_gpu, matriz_gris_f32_gpu, matriz_gris_u8_gpu)
    cuda.synchronize()

    return (matriz_gris_f32_gpu, matriz_gris_u8_gpu, matriz_sobel_u8_gpu, height, width, blockspergrid, THREADS_PER_BLOCK)