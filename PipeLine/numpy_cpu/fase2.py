import numpy as np

def procesar(matrizGris, config):
    try:
        # Aseguramos float32 para las operaciones matemáticas
        matrizGris_float = matrizGris.astype(np.float32)
        alto, ancho = matrizGris_float.shape

        # Inicializamos matrices de salida
        matrizGx = np.zeros((alto, ancho), dtype=np.float32)
        matrizGy = np.zeros((alto, ancho), dtype=np.float32)
        matrizSobel = np.zeros((alto, ancho), dtype=np.float32)

        # --- EXTRACCIÓN DE VECINOS (SLICING) ---
        p00 = matrizGris_float[:-2, :-2]
        p01 = matrizGris_float[:-2, 1:-1]
        p02 = matrizGris_float[:-2, 2:]
        
        p10 = matrizGris_float[1:-1, :-2]
        p12 = matrizGris_float[1:-1, 2:]
        
        p20 = matrizGris_float[2:, :-2]
        p21 = matrizGris_float[2:, 1:-1]
        p22 = matrizGris_float[2:, 2:]

        # --- OPERADORES EXPLÍCITOS ---
        KerGx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20)
        KerGy = (p00 + 2.0 * p01 + p02) - (p20 + 2.0 * p21 + p22)

        # --- ASIGNACIÓN Y MAGNITUD ---
        matrizGx[1:-1, 1:-1] = np.abs(KerGx)
        matrizGy[1:-1, 1:-1] = np.abs(KerGy)
        matrizSobel[1:-1, 1:-1] = np.sqrt(KerGx**2 + KerGy**2)

        # Recortamos a 255 y convertimos a uint8
        matrizGx = np.clip(matrizGx, 0, 255).astype(np.uint8)
        matrizGy = np.clip(matrizGy, 0, 255).astype(np.uint8)
        matrizSobel = np.clip(matrizSobel, 0, 255).astype(np.uint8)

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 2: {e}")

    # Retornamos tupla con las 4 matrices para mantener la compatibilidad
    return (matrizGris, matrizGx, matrizGy, matrizSobel)