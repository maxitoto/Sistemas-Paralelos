import argparse
import importlib
import time
import os
import sys
import csv
import json
import platform
import statistics

# --- CONFIGURACIÓN DE RUTAS ---
DIRECTORIO_RESULTADOS = "resultados" 
ARCHIVO_BASELINE = os.path.join(DIRECTORIO_RESULTADOS, "tiempos_base.json")
ARCHIVO_CSV = os.path.join(DIRECTORIO_RESULTADOS, "reporte_estadistico.csv")

if os.name == 'nt':
    # Obligamos a Numba a mirar en estas carpetas específicas
    os.environ['NUMBA_CUDA_DIR'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    os.environ['NUMBA_NVVM_LIBDIR'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\nvvm\bin"
    os.environ['NUMBA_LIBDEVICE_DIR'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\nvvm\libdevice"
    
    # Destrabamos la seguridad de Python para leer DLLs
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\nvvm\bin")

def parse_args():
    parser = argparse.ArgumentParser(description="Orquestador de Benchmarking")
    parser.add_argument('--contexto', type=str, default="Prueba de Rendimiento")
    parser.add_argument('--pipeline', type=str, required=True, help="Lista de pipelines separados por coma")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--save_baseline', action='store_true', help="Solo aplicable al secuencial: guarda el promedio como base")
    parser.add_argument('--runs', type=int, default=1, help="Número de ejecuciones para promediar")
    return parser.parse_args()

def obtener_info_hardware():
    return f"{platform.processor()} | {os.cpu_count()} Cores | {platform.system()}"

def guardar_baseline_total(input_name, t_total):
    datos = {}
    if os.path.exists(ARCHIVO_BASELINE):
        with open(ARCHIVO_BASELINE, 'r') as f:
            try: datos = json.load(f)
            except: pass
    datos[input_name] = {"total": t_total}
    with open(ARCHIVO_BASELINE, 'w') as f:
        json.dump(datos, f, indent=4)

def cargar_baseline_total(input_name):
    if not os.path.exists(ARCHIVO_BASELINE): return None
    with open(ARCHIVO_BASELINE, 'r') as f:
        try:
            datos = json.load(f).get(input_name, {})
            return datos.get('total')
        except: return None

# --- LÓGICA DE EXPORTACIÓN INTELIGENTE ---
def obtener_ultimo_contexto(ruta):
    if not os.path.exists(ruta) or os.path.getsize(ruta) == 0:
        return None
    ultimo_contexto = None
    with open(ruta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == "CONTEXTO DEL EXPERIMENTO:":
                ultimo_contexto = row[1]
    return ultimo_contexto

def exportar_csv(ruta, m, num_runs):
    ultimo_contexto = obtener_ultimo_contexto(ruta)
    contexto_actual = m['Contexto']
    necesita_encabezado = (ultimo_contexto != contexto_actual)
    
    with open(ruta, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if necesita_encabezado:
            if ultimo_contexto is not None:
                writer.writerow([])
                writer.writerow([])
                writer.writerow(["="*30, "="*30, "="*30]) 
                writer.writerow([])

            writer.writerow(["CONTEXTO DEL EXPERIMENTO:", contexto_actual])
            writer.writerow(["HARDWARE DETALLADO:", obtener_info_hardware()])
            writer.writerow(["METODOLOGÍA:", f"Promedio de {num_runs} iteraciones por método"])
            writer.writerow([])
            writer.writerow([
                "Método", "Workers", "T_Transf_IN(s)", "T_RGB_Promedio(s)", 
                "T_Sobel_Promedio(s)", "T_Total_Computo(s)", "T_Transf_OUT(s)", 
                "%_Blancos", "Speed-Up", "Performance(%)"
            ])

        writer.writerow([
            m['Metodo'], m['Workers'], 
            f"{m['T_IN']:.4f}", f"{m['T_RGB']:.4f}", f"{m['T_Sobel']:.4f}", 
            f"{m['T_Total']:.4f}", f"{m['T_OUT']:.4f}", 
            f"{m['P_Blancos']:.4f}", m['Speedup'], m['Performance']
        ])
# ------------------------------------------------

def ejecutar_pipeline(nombre_pipeline, args):
    print(f"\n>>> EXPERIMENTO: {nombre_pipeline.upper()} | Iteraciones: {args.runs}")
    
    fase_cero = None
    try:
        fase_cero = importlib.import_module(f"{nombre_pipeline}.fase0")
        print("    [Info] Fase 0 detectada (Se ejecutará sin contabilizar tiempo).")
    except ModuleNotFoundError:
        pass

    fases = []
    num_fase = 1
    while True:
        try:
            modulo = importlib.import_module(f"{nombre_pipeline}.fase{num_fase}")
            fases.append(modulo)
            num_fase += 1
        except ModuleNotFoundError:
            break

    historial_rgb = []
    historial_sobel = []
    historial_in = []
    historial_out = []
    p_blancos_final = 0.0

    for i in range(args.runs):
        datos_en_transito = args.input 
        t_rgb_corrida = 0.0
        t_sobel_corrida = 0.0
        t_in_corrida = 0.0
        t_out_corrida = 0.0

        if fase_cero:
            res_fase0 = fase_cero.procesar(datos_en_transito, (vars(args), nombre_pipeline))
            # Identifica de forma segura si devolvió (datos, tiempo)
            if isinstance(res_fase0, tuple) and len(res_fase0) == 2 and isinstance(res_fase0[1], float):
                datos_en_transito, t_in_corrida = res_fase0
            else:
                datos_en_transito = res_fase0

        for idx, modulo in enumerate(fases, start=1):
            inicio = time.perf_counter()
            res_modulo = modulo.procesar(datos_en_transito, (vars(args), nombre_pipeline))
            fin = time.perf_counter()
            
            duracion = fin - inicio
            
            if idx == 1:
                t_rgb_corrida = duracion
                datos_en_transito = res_modulo
            elif idx == 2:
                t_sobel_corrida = duracion
                datos_en_transito = res_modulo
            elif idx == 3:
                # Manejo dinámico: averigua si Fase3 pasó una tupla de (Blancos, Tiempo) o solo Blancos
                if isinstance(res_modulo, tuple) and len(res_modulo) == 2:
                    p_blancos_final, t_out_corrida = res_modulo
                else:
                    p_blancos_final = float(res_modulo)
                    t_out_corrida = 0.0

        historial_rgb.append(t_rgb_corrida)
        historial_sobel.append(t_sobel_corrida)
        historial_in.append(t_in_corrida)
        historial_out.append(t_out_corrida)
        
        print(f"    [Run {i+1}] Transf_IN: {t_in_corrida:.4f}s | T_Cómputo (F1+F2): {(t_rgb_corrida + t_sobel_corrida):.4f}s | Transf_OUT: {t_out_corrida:.4f}s")

    t_rgb_avg = statistics.mean(historial_rgb)
    t_sobel_avg = statistics.mean(historial_sobel)
    t_in_avg = statistics.mean(historial_in)
    t_out_avg = statistics.mean(historial_out)
    t_total_avg = t_rgb_avg + t_sobel_avg # Mantenemos el total puro para el Speedup

    return t_in_avg, t_rgb_avg, t_sobel_avg, t_total_avg, t_out_avg, p_blancos_final

def main():
    args = parse_args()
    os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)
    
    lista_pipelines = [p.strip() for p in args.pipeline.split(",")]

    for p_name in lista_pipelines:
        t_in, t_rgb, t_sobel, t_total, t_out, p_blancos = ejecutar_pipeline(p_name, args)

        if args.save_baseline and p_name.lower() == "secuencial":
            guardar_baseline_total(args.input, t_total)
            print(f" [!] NUEVO REFERENTE GUARDADO (Promedio de {args.runs} runs): {t_total:.4f}s")
        
        base_ref = cargar_baseline_total(args.input)

        def calcular_metricas(base, actual, workers):
            if not base: return "1.00x", "100.0%"
            if actual <= 0.0: actual = 1e-7  
            s = base / actual
            p = (s / workers) * 100
            return f"{s:.2f}x", f"{p:.1f}%"

        speedup, performance = calcular_metricas(base_ref, t_total, args.workers)

        metricas = {
            "Contexto": args.contexto, "Metodo": p_name, 
            "Workers": args.workers, 
            "T_IN": t_in, "T_RGB": t_rgb, "T_Sobel": t_sobel, 
            "T_Total": t_total, "T_OUT": t_out,
            "P_Blancos": p_blancos, "Speedup": speedup, "Performance": performance
        }

        exportar_csv(ARCHIVO_CSV, metricas, args.runs)
        print(f" [OK] {p_name} registrado. Speedup: {speedup}")

if __name__ == "__main__":
    main()