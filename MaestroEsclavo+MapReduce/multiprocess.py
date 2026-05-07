import time
import multiprocessing

USA_WORKERS = True

def ejecutar(datos, funcion_trabajo, workers):
    # Creamos el Pool clásico
    with multiprocessing.Pool(processes=workers) as pool:
        
        # --- ZONA DE CÓMPUTO Y SINCRONIZACIÓN ---
        inicio = time.perf_counter()
        
        # El .map de Pool ya es muy eficiente por defecto
        resultados = pool.map(funcion_trabajo, datos)
        
        fin = time.perf_counter()
        
    return resultados, (fin - inicio)