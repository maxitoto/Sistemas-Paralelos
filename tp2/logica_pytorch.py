# tp2/logica_pytorch.py
import math
import torch

COMPLEJIDAD = "O(n^3) - Acelerado con PyTorch (Tensores)"

def preparar_datos(datos_completos, workers):
    A_list, B_list = datos_completos
    # Convertimos las listas a Tensores de PyTorch
    A = torch.tensor(A_list, dtype=torch.int64)
    B = torch.tensor(B_list, dtype=torch.int64)
    
    chunk_size = math.ceil(len(A) / workers)
    if chunk_size == 0: chunk_size = 1
    return [(A[i:i + chunk_size], B) for i in range(0, len(A), chunk_size)]

def procesar_item(tarea):
    chunk_A, B = tarea 
    # Multiplicación matemática de tensores
    return torch.matmul(chunk_A, B)

def reducir_resultados(resultados_brutos):
    checksum = 0
    for bloque in resultados_brutos:
        # .item() extrae el número de Python desde dentro del Tensor C++
        checksum += int(torch.sum(bloque).item())
    return f"CHK-{checksum}"