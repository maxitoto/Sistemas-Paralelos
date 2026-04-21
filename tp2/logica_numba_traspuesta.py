# tp2/logica_numba_traspuesta.py
import math
import numpy as np
from numba import njit

COMPLEJIDAD = "O(n^3) - Numba + Optimización de Caché L1 (Fila x Fila)"

# El decorador compila esto a código máquina ultrarrápido
@njit
def multiplicar_traspuesta_rapido(chunk_A, BT):
    filas_A = chunk_A.shape[0]
    cols_A = chunk_A.shape[1]
    
    # Como BT está traspuesta, la cantidad de columnas originales de B
    # es ahora la cantidad de FILAS de BT.
    cols_B = BT.shape[0] 
    
    res = np.zeros((filas_A, cols_B), dtype=np.int64)
    
    for i in range(filas_A):
        for j in range(cols_B):
            suma = 0
            for k in range(cols_A):
                # ¡LA MAGIA ESTÁ AQUÍ! Ambos usan 'k' al final. 
                # Lectura contigua perfecta en memoria RAM.
                suma += chunk_A[i, k] * BT[j, k] 
            res[i, j] = suma
            
    return res

def preparar_datos(datos_completos, workers):
    A_list, BT_list = datos_completos
    # Convertimos a arreglos de 64 bits para Numba
    A = np.array(A_list, dtype=np.int64)
    BT = np.array(BT_list, dtype=np.int64)
    
    chunk_size = math.ceil(len(A) / workers)
    if chunk_size == 0: chunk_size = 1
    return [(A[i:i + chunk_size], BT) for i in range(0, len(A), chunk_size)]

def procesar_item(tarea):
    chunk_A, BT = tarea 
    return multiplicar_traspuesta_rapido(chunk_A, BT)

def reducir_resultados(resultados_brutos):
    checksum = 0
    for bloque in resultados_brutos:
        checksum += int(np.sum(bloque))
    return f"CHK-{checksum}"