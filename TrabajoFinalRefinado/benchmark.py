import importlib
import time
import os
import json # Agregado para leer la config en el pipeline

from utils.variables import obtener_variables
from utils.video import open_video, extract_audio, split_video, generar_lotes, merge_video, merge_audio, guardar_lote, clear_out
from utils.stats import stats, addResult, searchHardware, searchSoftware, init, exportar_csv

def ejecutar(work_path, temp_frames_origin_dir, temp_frames_filtered_dir, batch_size):
    tools_dir, work = work_path
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        modulo = importlib.import_module(f"{tools_dir}.{work}.process")
        pipeline_obj = modulo.Pipeline(config)
        
    except ModuleNotFoundError as e:
        print(f"❌ Error al cargar la herramienta '{work}': {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Inicialización de cronómetros
    t_total_disco = 0.0
    t_transfer_in = 0.0  
    t_computo1 = 0.0
    t_computo2 = 0.0
    t_transfer_out = 0.0
    
    pipeline_obj.calentar()

    generador = generar_lotes(temp_frames_origin_dir, batch_size)

    while True: 
        t_disco_init = time.perf_counter()
        try:
            lote_host, nombres_base, hay_lotes = next(generador)
        except StopIteration:
            break
        t_total_disco += (time.perf_counter() - t_disco_init)

        t_in_init = time.perf_counter()
        lote_dev, is_contable = pipeline_obj.host_to_device(lote_host)
        if is_contable: t_transfer_in += (time.perf_counter() - t_in_init)

        t_c1_init = time.perf_counter()
        lote_filtro1, is_contable = pipeline_obj.procesarComputo1(lote_dev)
        if is_contable: t_computo1 += (time.perf_counter() - t_c1_init)

        t_c2_init = time.perf_counter()
        lote_filtro2, is_contable = pipeline_obj.procesarComputo2(lote_filtro1)
        if is_contable: t_computo2 += (time.perf_counter() - t_c2_init)

        t_out_init = time.perf_counter()
        lote_final_host, is_contable = pipeline_obj.device_to_host(lote_filtro2)
        if is_contable: t_transfer_out += (time.perf_counter() - t_out_init)

        t_disco_init = time.perf_counter()
        guardar_lote(temp_frames_filtered_dir, lote_final_host, nombres_base)
        t_total_disco += (time.perf_counter() - t_disco_init)
        
        if not hay_lotes:
            break
            
    pipeline_obj.auxiliar()

    return t_total_disco, t_computo1, t_computo2, t_transfer_out, t_transfer_in

def main():
    input_dir, output_dir, temp_audio_dir, tools_dir, temp_video_mute_dir, temp_frames_origin_dir, temp_frames_filtered_dir, resultados_dir, batch_size, target_fps, codec_salida, name, time_baseline, baseline_method, pipeline, save_baseline = obtener_variables()

    #== ayuda de ia
    os.makedirs(temp_video_mute_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(temp_frames_origin_dir, exist_ok=True)
    os.makedirs(temp_frames_filtered_dir, exist_ok=True)
    archivo_video_in = os.path.join(input_dir, name)                     
    archivo_audio_temp = os.path.join(temp_audio_dir, "audio_original.aac") 
    archivo_video_mudo = os.path.join(temp_video_mute_dir, f"mudo_{name}")  
    archivo_video_final = os.path.join(output_dir, f"procesado_{name}")    
    #== ayuda de ia  

    video_ram = open_video(archivo_video_in)
    extract_audio(archivo_video_in, temp_audio_dir) 
    split_video(video_ram, temp_frames_origin_dir)

    hardware = searchHardware()
    software = searchSoftware()

    init(hardware, software, time_baseline, save_baseline)

    if save_baseline:
        pipeline = {"sec": baseline_method, **pipeline}

    for alias, ruta_modulo in pipeline.items():
        work_path = (tools_dir, ruta_modulo)
        
        t_total_disco, t_computo1, t_computo2, t_transfer_out, t_transfer_in = ejecutar(work_path, temp_frames_origin_dir, temp_frames_filtered_dir, batch_size)
        
        result = stats(t_total_disco, t_computo1, t_computo2, t_transfer_out, t_transfer_in, nombre_metodo=ruta_modulo)
        addResult(result)

    exportar_csv("static/res/reporte_estadistico.csv")

    merge_video(temp_frames_filtered_dir, archivo_video_mudo, codec_salida, target_fps)
    merge_audio(archivo_video_mudo, archivo_audio_temp, archivo_video_final)

    clear_out(temp_video_mute_dir)
    clear_out(temp_audio_dir)
    clear_out(temp_frames_origin_dir)
    clear_out(temp_frames_filtered_dir)


if __name__ == "__main__":
    main()


