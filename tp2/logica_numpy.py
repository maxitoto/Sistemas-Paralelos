# tp2/logica_numpy.py
import math
import numpy as np

COMPLEJIDAD = "O(n^3) - Acelerado con NumPy (C puro)"

def preparar_datos(datos_completos, workers):
    A_list, B_list = datos_completos
    # Convertimos las listas nativas a arreglos de NumPy de 64 bits
    A = np.array(A_list, dtype=np.int64)
    B = np.array(B_list, dtype=np.int64)
    
    chunk_size = math.ceil(len(A) / workers)
    if chunk_size == 0: chunk_size = 1
    return [(A[i:i + chunk_size], B) for i in range(0, len(A), chunk_size)]

def procesar_item(tarea):
    chunk_A, B = tarea 
    # El operador @ es el estándar de Python para multiplicar matrices en NumPy
    return chunk_A @ B

def reducir_resultados(resultados_brutos):
    checksum = 0
    for bloque in resultados_brutos:
        # np.sum() suma todo el bloque a la velocidad de la luz
        checksum += int(np.sum(bloque))
    return f"CHK-{checksum}"