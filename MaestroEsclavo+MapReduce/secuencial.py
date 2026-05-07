import time
USA_WORKERS = False
def ejecutar(datos, funcion_trabajo, _):
    inicio = time.perf_counter()
    res = [funcion_trabajo(x) for x in datos]
    return res, time.perf_counter() - inicio