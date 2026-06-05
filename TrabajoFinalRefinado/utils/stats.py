import os
import platform

try:
    from numba import cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

_config = {
    "hardware": "Desconocido",
    "software": "Desconocido",
    "time_baseline": 1.0,
    "save_baseline": False
}

_resultados = []

def searchHardware():
    """Detecta la CPU y busca la GPU usando Numba si está disponible."""
    cpu_info = f"CPU: {platform.processor()} | {os.cpu_count()} Cores"
    try:
        if HAS_CUDA and cuda.is_available():
            device = cuda.get_current_device()
            max_threads = getattr(device, 'MAX_THREADS_PER_BLOCK', 'N/A')
            nombre = device.name.decode('utf-8') if hasattr(device.name, 'decode') else device.name
            return f"{cpu_info} \nGPU: {nombre} | {max_threads} Threads/Block"
        else:
            return f"{cpu_info} \nGPU: No detectada"
    except Exception as e:
        return f"{cpu_info} \nError detectando GPU: {e}"

def searchSoftware():
    """Detecta el Sistema Operativo y la versión de Python."""
    return f"OS: {platform.system()} {platform.release()} | Python: {platform.python_version()}"

def init(hardware, software, time_baseline, save_baseline):
    """Inicializa la configuración base del módulo de estadísticas."""
    global _config, _resultados
    _config["hardware"] = hardware
    _config["software"] = software
    _config["time_baseline"] = float(time_baseline) if time_baseline else 1.0
    _config["save_baseline"] = save_baseline
    _resultados = [] # Limpiamos la lista al iniciar
    
    print(f"📊 [Stats] Inicializado. Baseline de referencia: {_config['time_baseline']}s")

def stats(t_total_disco, t_computo1, t_computo2, t_transfer_out, t_transfer_in, nombre_metodo):
    global _config
    
    t_computo_total = t_computo1 + t_computo2
    t_total_parcial = t_computo_total + t_total_disco
    t_global = t_total_parcial + t_transfer_in + t_transfer_out
    
    if _config["save_baseline"] and "secuencial" in nombre_metodo.lower():
        
        _config["time_baseline"] = t_global
        speedup = 1.0
        print(f"🔄 [Stats] ¡Nuevo Baseline Registrado en RAM!: {t_global:.4f}s")
        
    else:
        if _config["time_baseline"] and _config["time_baseline"] > 0 and t_global > 0:
            speedup = _config["time_baseline"] / t_global
        else:
            speedup = 0.0

    performance = speedup * 100 

    result = {
        "Metodo": nombre_metodo,
        "T_Perdido_Disco(s)": round(t_total_disco, 4),
        "T_Transf_IN(s)": round(t_transfer_in, 4),
        "T_Computo_1(s)": round(t_computo1, 4),
        "T_Computo_2(s)": round(t_computo2, 4),
        "T_Computo_Total(s)": round(t_computo_total, 4),
        "T_Transf_OUT(s)": round(t_transfer_out, 4),
        "T_Total_Global(s)": round(t_global, 4),
        "Speed-Up": f"{speedup:.2f}x",
        "Performance(%)": f"{performance:.1f}%"
    }
    
    return result

def addResult(result):
    """Guarda el resultado en la memoria y lo imprime por consola."""
    global _resultados
    _resultados.append(result)
    
    nombre = result['Metodo'].upper()
    t_glob = result['T_Total_Global(s)']
    sp_up = result['Speed-Up']
    print(f"🚀 [{nombre}] T_Global: {t_glob}s | Speed-Up: {sp_up}")

def exportar_csv(ruta_archivo):
    """Guarda o actualiza la lista de '_resultados' en un archivo físico."""
    global _resultados, _config
    import csv
    import os
    
    if not _resultados:
        print("⚠️ [Stats] No hay resultados para exportar.")
        return
        
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    
    # Comprobamos si el archivo ya existe ANTES de abrirlo
    archivo_existe = os.path.exists(ruta_archivo)
    
    # 'a' significa Append (Añadir al final de la hoja sin borrar nada)
    with open(ruta_archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Solo escribimos los títulos y encabezados si el archivo está en blanco
        if not archivo_existe:
            writer.writerow(["HARDWARE DETALLADO:", _config["hardware"].replace("\n", " | ")])
            writer.writerow(["SOFTWARE DETALLADO:", _config["software"]])
            writer.writerow([]) 
            columnas = list(_resultados[0].keys())
            writer.writerow(columnas)
        
        # Escribimos las filas de los resultados nuevos
        for res in _resultados:
            writer.writerow(list(res.values()))
            
    print(f"\n✅ [Stats] Reporte actualizado con éxito en: {ruta_archivo}")