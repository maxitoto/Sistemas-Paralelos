import time
from concurrent.futures import ProcessPoolExecutor

USA_WORKERS = True

def ejecutar(datos, funcion_trabajo, workers):
    # Instanciamos el pool (fuera del cronómetro)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        
        # Optimizamos el reparto de tareas (chunking) para que no sea tan lento
        tamano_paquete = max(1, len(datos) // (workers * 4))
        
        # --- ZONA DE CÓMPUTO Y SINCRONIZACIÓN ---
        inicio = time.perf_counter()
        
        # .map reparte y list() obliga a esperar a que todos terminen
        resultados = list(executor.map(funcion_trabajo, datos, chunksize=tamano_paquete))
        
        fin = time.perf_counter()
    
    return resultados, (fin - inicio)