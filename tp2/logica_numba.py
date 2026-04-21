# tp2/logica_numba.py
import math
import numpy as np
from numba import njit

COMPLEJIDAD = "O(n^3) - Compilado a Código Máquina (Numba LLVM)"

# El decorador @njit compila esta función exacta a código C puro
@njit(pararel = True)
def multiplicar_rapido(chunk_A, B):
    filas_A = chunk_A.shape[0]
    cols_A = chunk_A.shape[1]
    cols_B = B.shape[1]
    
    # Matriz vacía para guardar el resultado
    res = np.zeros((filas_A, cols_B), dtype=np.int64)
    
    for i in range(filas_A):
        for j in range(cols_B):
            suma = 0
            for k in range(cols_A):
                suma += chunk_A[i, k] * B[k, j]
            res[i, j] = suma
    return res

def preparar_datos(datos_completos, workers):
    A_list, B_list = datos_completos
    A = np.array(A_list, dtype=np.int64)
    B = np.array(B_list, dtype=np.int64)
    
    chunk_size = math.ceil(len(A) / workers)
    if chunk_size == 0: chunk_size = 1
    return [(A[i:i + chunk_size], B) for i in range(0, len(A), chunk_size)]

def procesar_item(tarea):
    chunk_A, B = tarea 
    return multiplicar_rapido(chunk_A, B)

def reducir_resultados(resultados_brutos):
    checksum = 0
    for bloque in resultados_brutos:
        checksum += int(np.sum(bloque))
    return f"CHK-{checksum}"