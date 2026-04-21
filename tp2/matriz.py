import random

# tp2/matriz.py
def obtener_datos(cant, orden, datos_manuales, seed):
    """Genera dos matrices cuadradas A y B de tamaño cant x cant."""
    if datos_manuales:
        # Si pasas datos manuales, el motor debería proveer una lógica para parsearlos
        # Para matrices, lo ideal es usar el generador aleatorio.
        return datos_manuales
        
    random.seed(seed)
    size = cant 
    min_val, max_val = orden
    
    A = [[random.randint(min_val, max_val) for _ in range(size)] for _ in range(size)]
    B = [[random.randint(min_val, max_val) for _ in range(size)] for _ in range(size)]
    
    return (A, B)