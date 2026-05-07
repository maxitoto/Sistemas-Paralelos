# tp1/logica.py

COMPLEJIDAD = "O(n)"

def preparar_datos(datos_completos, workers):
    """Corta el arreglo en porciones (chunks) equitativas para cada worker."""
    import math
    chunk_size = math.ceil(len(datos_completos) / workers)
    if chunk_size == 0: 
        chunk_size = 1
    return [datos_completos[i:i + chunk_size] for i in range(0, len(datos_completos), chunk_size)]

def procesar_item(chunk):
    """
    EL MOTOR BUSCA ESTE NOMBRE EXACTO. 
    Recibe un pedazo de arreglo y lo suma usando la función nativa en C.
    """
    return sum(chunk)

def reducir_resultados(resultados_brutos):
    """
    Junta todos los subtotales en un solo número final.
    """
    return sum(resultados_brutos)