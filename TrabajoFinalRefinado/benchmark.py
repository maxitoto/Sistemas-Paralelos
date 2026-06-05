import json
import os
import shutil
import importlib
from utils import video, audio, batch, memory, stadistics, outCsv, envinit

def limpiar_carpeta_filtrados(config):
    """
    Limpia los frames procesados por la herramienta anterior 
    para no mezclar resultados si se corre un perfil múltiple (ej. 'all').
    """
    filtered_dir = config["paths"]["temp_frames_filtered_dir"]
    if os.path.exists(filtered_dir):
        shutil.rmtree(filtered_dir)
    os.makedirs(filtered_dir)

def main():
    print("🚀 [Iniciando Benchmark HPC - Procesamiento de Video 4K]\n")

    # ==========================================
    # 1. CARGA DE CONFIGURACIÓN Y ENTORNO
    # ==========================================
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    envinit.limpiar_temporales(config)

    # ==========================================
    # 2. PRE-PROCESAMIENTO (I/O Aislado)
    # ==========================================
    audio.extraer_audio(config)
    cap, metadata = video.abrir_video(config)
    video.extraer_frames(config, cap, metadata["total_frames"])

    # ==========================================
    # 3. INICIALIZAR ESTADÍSTICAS (Comprobación Baseline)
    # ==========================================
    stats = stadistics.GestorEstadisticas(config, metadata)

    perfil_actual = "debug"
    herramientas = config["execution_profiles"][perfil_actual]

    # ==========================================
    # 4. BUCLE DE HERRAMIENTAS (El núcleo HPC)
    # ==========================================
    for alias, ruta_modulo in herramientas.items():
        print(f"\n⚙️ --- Evaluando Herramienta: {alias.upper()} ({ruta_modulo}) ---")
        
        limpiar_carpeta_filtrados(config)

        # Importación dinámica bajo el CONTRATO DE INTERFAZ
        try:

            modulo_tool = importlib.import_module(f"tools.{ruta_modulo}.process")
            pipeline = modulo_tool.Pipeline(config) 
            
        except Exception as e:
            print(f"⚠️ [Advertencia] Error al cargar 'tools.{ruta_modulo}.process'.\nDetalle: {e}")
            continue

        monitor = memory.MonitorMemoria()
        monitor.iniciar_registro()

        t_lectura_acum = 0.0
        t_computo_acum = 0.0
        t_escritura_acum = 0.0

        # --- FASE 0: WARM-UP ---
        # Se ejecuta ANTES de los cronómetros principales.
        # Enciende motores CUDA, compila JIT Numba o simplemente hace "pass" en CPU.
        print("🔥 Calentando motores (Fase 0)...")
        pipeline.calentar()

        # Obtenemos el generador de lotes
        generador_lotes = batch.procesar_en_lotes(config)

        # Bucle avanzado con control manual del iterador para cronometrar I/O pura
        while True:
            # --- MEDICIÓN DE LECTURA DE DISCO ---
            stats.tic("lectura")
            try:
                lote_matrices, nombres = next(generador_lotes)
                t_lectura_acum += stats.toc("lectura")
            except StopIteration:
                stats.toc("lectura") # Apaga el cronómetro si se acabó el video
                break

            # --- MEDICIÓN DE CÓMPUTO (Fases 1, 2 y 3) ---
            stats.tic("computo")
            
            # Fase 1: H2D (Host to Device) - Subir a VRAM
            print("🔥 Subiendo a VRAM (Fase 1)...")
            lote_dev = pipeline.host_to_device(lote_matrices)
            
            # Fase 2: Computo Puro (El Kernel / Filtro)
            print("🔥 Procesando (Fase 2)...")
            lote_res_dev = pipeline.procesar(lote_dev)
            
            # Fase 3: D2H (Device to Host) - Bajar a RAM
            print("🔥 Descargando de VRAM (Fase 3)...")
            lote_res_host = pipeline.device_to_host(lote_res_dev)
            
            t_computo_acum += stats.toc("computo")

            # Registrar pico en pleno apogeo de procesamiento
            monitor.actualizar_pico_ram()

            # --- MEDICIÓN DE ESCRITURA EN DISCO ---
            stats.tic("escritura")
            # El utils/batch.py se encarga de castear el float32 devuelto por Fase 3 a uint8
            batch.guardar_lote(config, lote_res_host, nombres)
            t_escritura_acum += stats.toc("escritura")

        print("Tratamientos Auxiliares (Fase 4)...")
        pipeline.auxiliar()

        # Al terminar el video, enviamos métricas detalladas a stadistics.py
        datos_memoria = monitor.obtener_resultados()
        stats.registrar_corrida(
            herramienta=alias,
            t_lectura=t_lectura_acum,
            t_computo=t_computo_acum,
            t_escritura=t_escritura_acum,
            memoria_stats=datos_memoria
        )

    # ==========================================
    # 5. POST-PROCESAMIENTO Y EXPORTACIÓN
    # ==========================================
    print("\n🎬 --- Ensamblado y Exportación ---")
    video.ensamblar_video(config)
    audio.fusionar_audio(config)

    head_csv, cuerpo_csv = stats.obtener_datos_exportacion()
    ruta_reporte = os.path.join(config["paths"]["resultados_dir"], "reporte_benchmark.csv")
    outCsv.guardar_reporte(ruta_reporte, head_csv, cuerpo_csv)

    envinit.limpiar_temporales(config)
    print("\n✅ [BENCHMARK COMPLETADO EXITOSAMENTE]")

if __name__ == "__main__":
    main()