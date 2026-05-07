#convolucion secuencial y magnitud gradiente (magnitud de sobel)

import math

def procesar(datos, config):
    
    matrizGris, alto, ancho  = datos
    
    matrizGx = [[0 for _ in range(ancho)] for _ in range(alto)]
    matrizGy = [[0 for _ in range(ancho)] for _ in range(alto)] 
    matrizSobel = [[0 for _ in range(ancho)] for _ in range(alto)] 


    #ahora defino los Kernels (ventanas para filtro sobel)

    Gx = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ]

    Gy = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1] ]

    # debemos ignorar los bordes extremos de la imagen gris, porque produce desbordamineto al queres mover el kerner por toda la imagen
    for fil in range(1, alto-1):
        for col in range(1, ancho-1):
            
            KerGx = 0
            KerGy = 0

            #3x3 alrededor del píxel central (y, x)
            for ky in range(-1 , 2):
                for kx in range(-1, 2):
                    
                    pixel = float(matrizGris[fil+ky][col+kx]) #obtengo el pixel

                    KerGx += Gx[ky+1][kx+1] * pixel #obtengo el pixel multiplicado por el kernel x
                    KerGy += Gy[ky+1][kx+1] * pixel #obtengo el pixel multiplicado por el kernel y

            abs_gx = abs(KerGx)
            matrizGx[fil][col] = int(min(abs_gx, 255))
    
            abs_gy = abs(KerGy)
            matrizGy[fil][col] = int(min(abs_gy, 255))

            magnitud = math.sqrt(KerGx**2 + KerGy**2) #paso 4, pero mejor lo incorporo aquí
            matrizSobel[fil][col] = int(min(magnitud, 255))

    return (matrizGris, matrizGx, matrizGy, matrizSobel)