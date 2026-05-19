from PIL import Image as pil
import numpy as np
import os
from numba import cuda
import math

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

# fase 1 encontrar la imagen en paralelo con numba.
def procesar(imagen_path, config):

    if (not os.path.exists(imagen_path)):
        raise FileExistsError(f"Error: La imagen no fue encontrada en la ruta indicada \n {imagen_path}")
    
    try:
        imagen_ram = pil.open(imagen_path)
        
        #imagen_grises = imagen_pil.convert('L') # pillow tiene una funcion para llevar la imagen a escala de grises (no lo voy a usar)
        
        # Llevamos la imagen a una matriz 3D (Alto x Ancho x 3 (Red,Green,Blue)) en float32 para cálculos
        aux_matriz_rgb_cpu = np.asarray(imagen_ram, dtype=np.float32)
        aux_height, aux_width = aux_matriz_rgb_cpu.shape

        aux_matriz_rgb_gpu = cuda.to_device(aux_matriz_rgb_cpu) 


        #Resultados
        aux_matriz_gris_gpu = cuda.device_array((aux_height, aux_width), dtype=np.uint8)
        aux_matriz_sobel_gpu = cuda.device_array((aux_height, aux_width), dtype=np.uint8)

        AUX_THREADS_PER_BLOCK = (16, 16) # esto se tiene que ir modificando dependiendo mi GPU, pueden ser 8x8, 16x16, 8x16, 16x8, etc.

        aux_blockspergrid_y = (aux_height + AUX_THREADS_PER_BLOCK[0] - 1) // AUX_THREADS_PER_BLOCK[0]
        aux_blockspergrid_x = (aux_width + AUX_THREADS_PER_BLOCK[1] - 1) // AUX_THREADS_PER_BLOCK[1]
        aux_blockspergrid = (aux_blockspergrid_y, aux_blockspergrid_x)

        # Warmup
        rgb_to_gray_cuda[aux_blockspergrid, AUX_THREADS_PER_BLOCK](aux_matriz_rgb_gpu, aux_matriz_gris_gpu)
        sobel_cuda[aux_blockspergrid, AUX_THREADS_PER_BLOCK](aux_matriz_gris_gpu, aux_matriz_sobel_gpu)
        cuda.synchronize()

        #recalculamos todo lo enviamos
        matriz_rgb_cpu = np.asarray(imagen_ram, dtype=np.float32)
        height, width = matriz_rgb_cpu.shape
        matriz_rgb_gpu = cuda.to_device(matriz_rgb_cpu)
        matriz_gris_gpu = cuda.device_array((height, width), dtype=np.uint8)
        matriz_sobel_gpu = cuda.device_array((height, width), dtype=np.uint8)
        THREADS_PER_BLOCK = (16, 16) # esto se tiene que ir modificando dependiendo mi GPU, pueden ser 8x8, 16x16, 8x16, 16x8, etc.

        blockspergrid_y = (height + THREADS_PER_BLOCK[0] - 1) // THREADS_PER_BLOCK[0]
        blockspergrid_x = (width + THREADS_PER_BLOCK[1] - 1) // THREADS_PER_BLOCK[1]
        blockspergrid = (blockspergrid_y, blockspergrid_x)
        
    except Exception as e:
        raise ValueError(f"Falla en Fase 0: No se pudo procesar el archivo. Error: {e}")
    
    return (matriz_rgb_gpu, matriz_gris_gpu, matriz_sobel_gpu, height, width, blockspergrid, THREADS_PER_BLOCK)


@cuda.jit
def rgb_to_gray_cuda(rgb: np.ndarray, gray: np.ndarray) -> None:

    # Cada hilo obtiene una coordenada (y, x) unica en una grilla 2D.
    y, x = cuda.grid(2)

    # Un hilo procesa un solo pixel: no hay for en Python porque
    # el recorrido completo lo hacen miles de hilos en paralelo.
    r = float(rgb[y, x, 0]) # x,y = [r,g,b]
    g = float(rgb[y, x, 1])
    b = float(rgb[y, x, 2])

    i = int(0.299 * r + 0.587 * g + 0.114 * b)  # (r+g+b)/3 = promedio == gris
    #i = 0 if i < 0 else 255 if i > 255 else i   # chequea que no se vaya de rango (0,255)

    # i tiene el valor gris del pixel
    gray[y, x] = i

@cuda.jit
def sobel_cuda(gray: np.ndarray, out: np.ndarray) -> None:

    # Igual que en rgb_to_gray_cuda: cada hilo trabaja sobre un pixel (y, x).
    y, x = cuda.grid(2)

    # iniciamos la gradiente en cero
    gx = 0
    gy = 0

    # Estos for NO recorren toda la imagen: solo la vecindad 3x3
    # del pixel actual asignado a este hilo.
    for ky in range(3):
        for kx in range(3):
            p = int(gray[y + ky - 1, x + kx - 1])
            gx += p * gx_kernel[ky][kx]
            gy += p * gy_kernel[ky][kx]

    mag = int(math.sqrt(gx * gx + gy * gy))
    out[y, x] = 255 if mag > 255 else mag

    total_pixels = gray.size
    white_pixels = np.count_nonzero(gray == 255)
    return float(white_pixels) * 100.0 / float(total_pixels)