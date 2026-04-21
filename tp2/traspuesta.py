#tp2/traspuesta.py
import random

def obtener_datos(cant, orden, datos_manuales, seed):
    """Genera matrices A y B, pero devuelve B ya traspuesta."""
    random.seed(seed)
    size = cant
    min_val, max_val = orden
    
    A = [[random.randint(min_val, max_val) for _ in range(size)] for _ in range(size)]
    B_original = [[random.randint(min_val, max_val) for _ in range(size)] for _ in range(size)]
    
    # Traspuesta inmediata: Intercambiamos filas por columnas
    B_T = [[B_original[j][i] for j in range(size)] for i in range(size)]
    
    return (A, B_T)