import random

def obtener_datos(cant, orden, datos_manuales, seed):
    """
    Genera los datos usando la semilla recibida del motor.
    """
    # Si hay datos manuales, los usamos y listo
    if datos_manuales and len(datos_manuales) > 0:
        return datos_manuales
        
    # Seteamos la semilla recibida
    random.seed(seed)
    
    # Calculamos el tamaño (10^n)
    tamaño = 10 ** cant

    # orden o costo de computo
    min = orden[0]
    max = orden[1]
    
    # Usamos el rango del profesor (0 a 50)
    return [random.randint(min, max) for _ in range(tamaño)]