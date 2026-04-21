import numpy as np

def obtener_datos(cant, orden, datos_manuales, seed):
    """
    Genera los datos usando NumPy y la semilla recibida del motor.
    """        
    # Seteamos la semilla en el motor de NumPy
    np.random.seed(seed)
    
    # Calculamos el tamaño (10^n)
    tamaño = 10 ** cant
    
    # 1. np.random.rand(tamaño) crea un arreglo de floats entre 0.0 y 0.999...
    # 2. Multiplicamos por la amplitud del rango y sumamos el mínimo
    # 3. .astype(np.int64) recorta los decimales para dejar enteros limpios
    arreglo_floats = np.random.rand(tamaño)
    
    return arreglo_floats