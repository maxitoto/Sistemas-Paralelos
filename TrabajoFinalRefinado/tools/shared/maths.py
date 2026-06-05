import numpy as np

def filterGris(frame_float32):
    """
    Convierte un frame RGB (float32) a Escala de Grises (Luminancia).
    Retorna una matriz 2D (Alto, Ancho) en float32.
    """
    # Pesos estándar para la percepción del ojo humano (R, G, B)
    pesos = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    
    # El producto punto colapsa el último eje (los 3 canales) en 1 solo
    frame_gris = np.dot(frame_float32[..., :3], pesos)
    
    return frame_gris