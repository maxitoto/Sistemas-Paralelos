# tp0/logica.py
import math
import sys

# --- BYPASS DE SEGURIDAD PARA PYTHON 3.11+ ---
# Permite convertir enteros gigantes (como factoriales de 50000) a texto sin crashear.
try:
    sys.set_int_max_str_digits(1000000)
except AttributeError:
    pass # Si tienes una versión de Python antigua, ignora esto.

COMPLEJIDAD = "O(n)"

def preparar_datos(datos_completos, workers):
    """Corta los datos en porciones (chunks) equitativas para cada worker."""
    chunk_size = math.ceil(len(datos_completos) / workers)
    if chunk_size == 0: 
        chunk_size = 1
    return [datos_completos[i:i + chunk_size] for i in range(0, len(datos_completos), chunk_size)]

def procesar_item(chunk):
    """
    Factoriales en paralelo:
    Recibe un chunk (lista de números) y devuelve una lista 
    con el factorial de cada uno de esos números.
    """
    return [math.factorial(e) for e in chunk]

def reducir_resultados(resultados_brutos):
    """
    Promedio en secuencial:
    Junta todas las listas de factoriales de los workers, 
    las aplana en una sola lista y calcula el promedio.
    """
    # 1. Aplanar la lista de listas usando comprensión de listas (más rápido en C que el for)
    lista_plana = [item for sublista in resultados_brutos for item in sublista]
        
    # 2. Calcular el promedio secuencial de todos los factoriales
    if len(lista_plana) == 0:
        return 0
        
    suma_total = sum(lista_plana)
    promedio = suma_total // len(lista_plana)

    # 3. Formateo científico seguro
    str_promedio = str(promedio)
    if len(str_promedio) > 10:
        # Usa el 1er dígito, una COMA, 2 decimales [1:3], y la cantidad de ceros
        simplificado = f"{str_promedio[0]},{str_promedio[1:3]}e+{len(str_promedio)-1}"
        return simplificado
    
    return promedio