import time
from concurrent.futures import ThreadPoolExecutor
USA_WORKERS = True
def ejecutar(datos, funcion_trabajo, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inicio = time.perf_counter()
        res = list(executor.map(funcion_trabajo, datos))
        t = time.perf_counter() - inicio
    return res, t