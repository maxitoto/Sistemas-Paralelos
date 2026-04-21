import time
import threading
from queue import Queue

USA_WORKERS = True

# Esta es la función que corre cada hilo de forma independiente
def worker(task_queue, results, funcion_trabajo):
    while True:
        # El hilo se queda acá bloqueado esperando que aparezca algo en la cola
        item = task_queue.get()
        
        # Si recibe None, es la señal de que debe morir (terminar el bucle)
        if item is None:
            break
        
        index, valor = item
        # Ejecuta la lógica y guarda el resultado en su posición correspondiente
        results[index] = funcion_trabajo(valor)
        
        # Le avisa a la cola que la tarea ya se completó
        task_queue.task_done()

def ejecutar(datos, funcion_trabajo, workers):
    task_queue = Queue()
    results = [None] * len(datos)
    
    # 1. BUROCRACIA: Lanzamos los hilos (fuera del cronómetro)
    threads = []
    for _ in range(workers):
        t = threading.Thread(target=worker, args=(task_queue, results, funcion_trabajo))
        t.daemon = True # Para que no traben el programa si algo falla
        t.start()
        threads.append(t)

    # 2. ZONA DE CÓMPUTO Y SINCRONIZACIÓN
    inicio = time.perf_counter()
    
    # Repartimos el trabajo cargando la cola
    for i, x in enumerate(datos):
        task_queue.put((i, x))
    
    # El método .join() de la Queue bloquea el programa hasta que la cola esté vacía
    # Esto garantiza la sincronización final
    task_queue.join()
    
    fin = time.perf_counter()

    # 3. LIMPIEZA: Matamos a los hilos mandando señales de fin
    for _ in range(workers):
        task_queue.put(None)
    for t in threads:
        t.join()

    return results, (fin - inicio)