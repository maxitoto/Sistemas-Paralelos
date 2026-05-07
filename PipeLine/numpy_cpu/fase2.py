import numpy as np

def procesar(datos_entrada, config):

    # Aseguramos que la imagen base sea float32 para las operaciones matemáticas
    matrizGris = datos_entrada.astype(np.float32)
    alto, ancho = matrizGris.shape

    # Inicializamos las matrices de salida
    matrizGx = np.zeros((alto, ancho), dtype=np.float32)
    matrizGy = np.zeros((alto, ancho), dtype=np.float32)
    matrizSobel = np.zeros((alto, ancho), dtype=np.float32)

    # --- EXTRACCIÓN DE VECINOS VECTORIZADA ---
    # Recortamos la matriz original para simular el desplazamiento en las 8 direcciones
    p00 = matrizGris[:-2, :-2]   # Arriba-Izquierda
    p01 = matrizGris[:-2, 1:-1]  # Arriba-Centro
    p02 = matrizGris[:-2, 2:]    # Arriba-Derecha
    
    p10 = matrizGris[1:-1, :-2]  # Centro-Izquierda
    p12 = matrizGris[1:-1, 2:]   # Centro-Derecha
    
    p20 = matrizGris[2:, :-2]    # Abajo-Izquierda
    p21 = matrizGris[2:, 1:-1]   # Abajo-Centro
    p22 = matrizGris[2:, 2:]     # Abajo-Derecha

    # --- OPERADORES EXPLÍCITOS ---
    # Aplicamos la matemática de los kernels a toda la imagen simultáneamente
    
    # Gx: Derivada horizontal (basado en tu kernel original)
    KerGx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20)
    
    # Gy: Derivada vertical (basado en tu kernel original: [1, 2, 1] arriba, [-1, -2, -1] abajo)
    KerGy = (p00 + 2.0 * p01 + p02) - (p20 + 2.0 * p21 + p22)

    # --- ASIGNACIÓN Y MAGNITUD ---
    # Guardamos los resultados respetando el borde de 1 píxel (1:-1)
    matrizGx[1:-1, 1:-1] = np.abs(KerGx)
    matrizGy[1:-1, 1:-1] = np.abs(KerGy)
    
    # Raíz cuadrada vectorizada para la magnitud de Sobel
    matrizSobel[1:-1, 1:-1] = np.sqrt(KerGx**2 + KerGy**2)

    # --- CONVERSIÓN FINAL PARA FASE 3 ---
    # Recortamos a 255 y convertimos a uint8
    matrizGris = matrizGris.astype(np.uint8)
    matrizGx = np.clip(matrizGx, 0, 255).astype(np.uint8)
    matrizGy = np.clip(matrizGy, 0, 255).astype(np.uint8)
    matrizSobel = np.clip(matrizSobel, 0, 255).astype(np.uint8)

    return (matrizGris, matrizGx, matrizGy, matrizSobel)