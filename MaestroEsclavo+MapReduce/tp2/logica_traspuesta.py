# tp2/logica_traspuesta.py
import math

COMPLEJIDAD = "O(n^3) - Optimizada por Caché"

def preparar_datos(datos_completos, workers):
    A, B_T = datos_completos 
    chunk_size = math.ceil(len(A) / workers)
    if chunk_size == 0: chunk_size = 1
    return [(A[i:i + chunk_size], B_T) for i in range(0, len(A), chunk_size)]

def procesar_item(tarea):
    chunk_A, BT = tarea 
    res_parcial = []
    
    for fila_a in chunk_A:
        nueva_fila = []
        for fila_bt in BT:
            # MULTIPLICACIÓN RÁPIDA: Fila x Fila (Contigua en memoria)
            punto = sum(fila_a[k] * fila_bt[k] for k in range(len(fila_a)))
            nueva_fila.append(punto)
        res_parcial.append(nueva_fila)
    return res_parcial

def reducir_resultados(resultados_brutos):
    """
    Toma los bloques procesados por los workers y, en lugar de armar la matriz, 
    calcula la suma total de todos los elementos como un Checksum de verificación.
    """
    checksum = 0
    
    # resultados_brutos es una lista de bloques. Cada bloque tiene varias filas.
    for bloque in resultados_brutos:
        for fila in bloque:
            checksum += sum(fila)
            
    return f"CHK-{checksum}"