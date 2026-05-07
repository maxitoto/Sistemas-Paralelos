import argparse
import importlib
import multiprocessing
import sys
import os
import subprocess
import csv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Motor de Benchmarking: Hilos vs Procesos",
        epilog="Ejemplo: python benchmark.py --desde tp2 --datos matriz --cant 300 --runners secuencial process --sort speedup tiempo --desc"
    )
    
    parser.add_argument('--contexto', type=str, default="Benchmarking", help="Descripción de la tarea")
    parser.add_argument('--desde', type=str, required=True, help="Carpeta del TP (ej: tp0, tp1)")
    parser.add_argument('--csv', type=str, help="Ruta del CSV para exportar (ej: reporte.csv)")
    
    parser.add_argument('--seed', type=int, default=2026, help="Semilla (def: 2026)")
    parser.add_argument('--datos', default='arreglo', help="Módulo de datos")
    parser.add_argument('--cant', type=int, default=1, help="Potencia de tamaño de datos")
    parser.add_argument('--orden', nargs=2, type=int, default=[1, 50], help="Limites de costo de computo")
    parser.add_argument('--input', nargs='*', type=int, default=[], help="Datos manuales")
    parser.add_argument('--logica', type=str, default='logica', help="Módulo de lógica a utilizar")

    parser.add_argument('--runners', nargs='*', default=[], help="Runners base (ej: secuencial threads process)")
    parser.add_argument('--workers', nargs='*', type=int, default=[1, 2, 4], help="Workers")
    
    # --- NUEVOS ARGUMENTOS DE ORDENAMIENTO ---
    parser.add_argument('--sort', nargs='+', choices=['tiempo', 'speedup', 'eficiencia', 'tipo', 'workers'], 
                        help="Criterios para ordenar la tabla final (ej: --sort speedup tiempo)")
    parser.add_argument('--desc', action='store_true', help="Aplica orden descendente (mayor a menor)")
    
    return parser.parse_args()

def obtener_cores():
    logical = os.cpu_count()
    physical = logical
    try:
        if sys.platform == 'darwin':
            physical = int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).decode().strip())
        elif sys.platform == 'win32':
            out = subprocess.check_output(['wmic', 'cpu', 'get', 'NumberOfCores']).decode()
            physical = int(out.split()[1])
    except:
        pass
    return physical, logical

def cargar_datos_y_entorno(args):
    """Carga los módulos dinámicos y genera los datos requeridos."""
    try:
        mod_datos = importlib.import_module(f"{args.desde}.{args.datos}")
        datos_crudos = mod_datos.obtener_datos(args.cant, args.orden, args.input, args.seed)
        
        if args.input:
            esfuerzo_txt = "Datos ingresados manualmente"
        else:
            esfuerzo_txt = f"Aleatorio entre {args.orden[0]} y {args.orden[1]}"
            
        mod_logica = importlib.import_module(f"{args.desde}.{args.logica}")
        cores_f, cores_l = obtener_cores()
        
        return datos_crudos, esfuerzo_txt, mod_logica, cores_f, cores_l
    
    except Exception as e:
        print(f"[!] Error al cargar módulos o datos: {e}")
        sys.exit(1)

def mostrar_encabezado_experimento(contexto, cores_f, cores_l, esfuerzo_txt, cantidad_elementos):
    """Imprime la cabecera de información en la terminal."""
    print("="*60)
    print(" INFO DE EXPERIMENTO")
    print("="*60)
    print(f" Contexto: {contexto}")
    print(f" Hardware: {cores_f} Cores físicos / {cores_l} Lógicos")
    print(f" Esfuerzo de computo: {esfuerzo_txt}")
    print(f" Cantidad de elementos: {cantidad_elementos}")
    print("="*60)

def imprimir_tabla(resultados, titulo):
    """Imprime una tabla formateada a partir de una lista de diccionarios de resultados."""
    print(f"\n>>> {titulo} <<<")
    header = f"{'Tipo':<15} | {'Workers':<7} | {'Tiempo (s)':<12} | {'Speed-up':<8} | {'Eficiencia':<10} | {'Resultado'}"
    print(header)
    print("-" * len(header))
    
    for f in resultados:
        s_str = f"{f['SPEEDUP']:.2f}x"
        e_str = f"{f['EFICIENCIA_PCT']:.1f}%"
        res_str = f['RESULTADO']
        res_print = res_str if len(res_str) <= 15 else res_str[:12] + "..."
        print(f"{f['TIPO']:<15} | {f['WORKERS']:<7} | {f['TIEMPO_S']:.9f} | {s_str:<8} | {e_str:<10} | {res_print}")

def ejecutar_bateria_pruebas(runners, workers_list, datos_crudos, mod_logica):
    """Itera sobre los runners, imprime en vivo y recolecta métricas."""
    print(f"\n>>> EJECUCIÓN EN TIEMPO REAL (Orden de llegada) <<<")
    header = f"{'Tipo':<15} | {'Workers':<7} | {'Tiempo (s)':<12} | {'Speed-up':<8} | {'Eficiencia':<10} | {'Resultado'}"
    print(header)
    print("-" * len(header))
    
    t_referencia = None
    resultados_recolectados = [] 

    for runner_name in runners:
        try:
            modulo = importlib.import_module(runner_name)
            usa_w = getattr(modulo, 'USA_WORKERS', True)
            actual_workers = workers_list if usa_w else [1]

            for w in actual_workers:
                if hasattr(mod_logica, 'preparar_datos'):
                    datos_listos = mod_logica.preparar_datos(datos_crudos, w)
                else:
                    datos_listos = datos_crudos 

                res_brutos, t_n = modulo.ejecutar(datos_listos, mod_logica.procesar_item, w)
                
                if hasattr(mod_logica, 'reducir_resultados'):
                    res_final = mod_logica.reducir_resultados(res_brutos)
                else:
                    res_final = res_brutos 

                if t_referencia is None:
                    t_referencia = t_n 
                
                speedup = t_referencia / t_n if t_referencia else 0
                eficiencia = speedup / w if w > 0 else 0

                s_str = f"{speedup:.2f}x"
                e_str = f"{eficiencia*100:.1f}%"
                res_str = str(res_final)
                res_print = res_str if len(res_str) <= 15 else res_str[:12] + "..."

                # Imprimir la fila en vivo
                print(f"{runner_name:<15} | {w:<7} | {t_n:.9f} | {s_str:<8} | {e_str:<10} | {res_print}")

                resultados_recolectados.append({
                    'TIPO': runner_name,
                    'WORKERS': w,
                    'TIEMPO_S': round(t_n, 9),
                    'SPEEDUP': round(speedup, 2),
                    'EFICIENCIA_PCT': round(eficiencia * 100, 1),
                    'RESULTADO': res_str
                })
                
        except ModuleNotFoundError:
            print(f"[!] Error: No se encontró el runner '{runner_name}.py'")

    return resultados_recolectados

def exportar_reporte_csv(ruta_csv, resultados, contexto, cores_f, cores_l, esfuerzo_txt, cant_elementos):
    """Guarda la info del experimento y los resultados en un CSV."""
    if not ruta_csv:
        return
        
    try:
        with open(ruta_csv, mode='w', newline='', encoding='utf-8') as archivo:
            archivo.write(f"# EXPERIMENTO: {contexto}\n")
            archivo.write(f"# HARDWARE: {cores_f} F / {cores_l} L\n")
            archivo.write(f"# ESFUERZO: {esfuerzo_txt}\n")
            archivo.write(f"# ELEMENTOS: {cant_elementos}\n")
            archivo.write("# \n") 
            
            campos = ['TIPO', 'WORKERS', 'TIEMPO_S', 'SPEEDUP', 'EFICIENCIA_PCT', 'RESULTADO']
            escritor = csv.DictWriter(archivo, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(resultados)
            
        print(f"\n[+] Reporte exportado a: {ruta_csv}")
    except Exception as e:
        print(f"\n[-] Error al escribir el CSV: {e}")

def main():
    args = parse_args()
    
    if args.csv and not os.path.dirname(args.csv):
        args.csv = os.path.join(args.desde, args.csv)
                
    # 1. Carga
    datos_crudos, esfuerzo_txt, mod_logica, cores_f, cores_l = cargar_datos_y_entorno(args)

    # 2. Cabecera en terminal
    mostrar_encabezado_experimento(args.contexto, cores_f, cores_l, esfuerzo_txt, len(datos_crudos))

    # 3. Ejecución y Tabla en Vivo
    datos_recolectados = ejecutar_bateria_pruebas(args.runners, args.workers, datos_crudos, mod_logica)

    # 4. Lógica de Ordenamiento (NUEVO)
    if args.sort:
        # Diccionario para traducir los argumentos de la terminal a las claves del diccionario
        mapa_sort = {
            'tiempo': 'TIEMPO_S',
            'speedup': 'SPEEDUP',
            'eficiencia': 'EFICIENCIA_PCT',
            'tipo': 'TIPO',
            'workers': 'WORKERS'
        }
        # Extraemos las claves por las que el usuario quiere ordenar
        claves_orden = [mapa_sort[k] for k in args.sort]
        
        # Ordenamos la lista in-place. Si hay múltiples claves, Python ordena por la 1ra, luego desempata con la 2da, etc.
        datos_recolectados.sort(key=lambda x: tuple(x[k] for k in claves_orden), reverse=args.desc)
        
        # Imprimimos la tabla ordenada
        texto_orden = " -> ".join(args.sort).upper()
        if args.desc: texto_orden += " (DESCENDENTE)"
        imprimir_tabla(datos_recolectados, f"TABLA ORDENADA POR: {texto_orden}")

    # 5. Exportación
    exportar_reporte_csv(
        args.csv, 
        datos_recolectados, 
        args.contexto, 
        cores_f, 
        cores_l, 
        esfuerzo_txt, 
        len(datos_crudos)
    )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'help':
        sys.argv[1] = '--help'
    main()