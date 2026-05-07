#convolucion secuencial y magnitud gradiente (magnitud de sobel)

import math

def procesar(datos, config):
    
    matrizGris, alto, ancho = datos
    
    matrizGx = [[0 for _ in range(ancho)] for _ in range(alto)]
    matrizGy = [[0 for _ in range(ancho)] for _ in range(alto)] 
    matrizSobel = [[0 for _ in range(ancho)] for _ in range(alto)] 

    # calcula 1D y trasponer en el mismo paso
    def aplicar_kernel_y_trasponer(matriz, h, w, kernel):
        resultado = [[0.0 for _ in range(h)] for _ in range(w)]
        for f in range(h):
            for c in range(1, w - 1):
                resultado[c][f] = float(matriz[f][c-1] * kernel[0] + 
                                        matriz[f][c] * kernel[1] + 
                                        matriz[f][c+1] * kernel[2])
        return resultado

    # ahora defino los Kernels separables a 1 dimension
    K_derivada = [-1, 0, 1]
    K_suavizado = [1, 2, 1]

    # Pasadas horizontales y verticales integrando la traspuesta
    gx_h_t = aplicar_kernel_y_trasponer(matrizGris, alto, ancho, K_derivada)
    gx_v = aplicar_kernel_y_trasponer(gx_h_t, ancho, alto, K_suavizado)

    gy_h_t = aplicar_kernel_y_trasponer(matrizGris, alto, ancho, K_suavizado)
    gy_v = aplicar_kernel_y_trasponer(gy_h_t, ancho, alto, K_derivada)

    # debemos ignorar los bordes extremos de la imagen gris, porque produce desbordamineto al queres mover el kerner por toda la imagen
    for fil in range(1, alto-1):
        for col in range(1, ancho-1):
            
            # recuperamos los valores ya calculados de las matrices separadas
            KerGx = gx_v[fil][col]
            KerGy = gy_v[fil][col]

            abs_gx = abs(KerGx)
            matrizGx[fil][col] = int(min(abs_gx, 255))
    
            abs_gy = abs(KerGy)
            matrizGy[fil][col] = int(min(abs_gy, 255))

            magnitud = math.sqrt(KerGx**2 + KerGy**2) #paso 4, pero mejor lo incorporo aquí
            matrizSobel[fil][col] = int(min(magnitud, 255))

    return (matrizGris, matrizGx, matrizGy, matrizSobel)