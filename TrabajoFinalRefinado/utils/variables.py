import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=str, required=True, default="debug", help="Tipo de ejecuciones")
    parser.add_argument("--save_baseline", action="store_true", help='Guardar el tiempo secuencial como base')
    parser.add_argument("--save_estimate_baseline", action="store_true", help='Guardar el tiempo estimado como base usando solo un frame (1)')
    return parser.parse_args()

def obtener_variables():
    try:
        config = json.load(open("config.json", "r"))
    except:
        print("No se pudo leer el archivo de configuración")
        exit()

    args = parse_args()

    if os.name == 'nt':
        # Obligamos a Numba a mirar en estas carpetas específicas
        os.environ['NUMBA_CUDA_DIR'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
        os.environ['NUMBA_NVVM_LIBDIR'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\nvvm\bin"
        os.environ['NUMBA_LIBDEVICE_DIR'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\nvvm\libdevice"
    
        # Destrabamos la seguridad de Python para leer DLLs
        os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
        os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\nvvm\bin")

    paths = config["paths"]
    video_settings = config["video_settings"]
    bechmark_settings = config["benchmark_settings"]

    input_dir = paths["input_dir"]
    output_dir = paths["output_dir"]
    temp_audio_dir = paths["temp_audio_dir"]
    temp_frames_origin_dir = paths["temp_frames_origin_dir"]
    temp_frames_filtered_dir = paths["temp_frames_filtered_dir"]
    temp_video_mute_dir = paths["temp_video_muted_dir"]
    resultados_dir = paths["resultados_dir"]
    tools_dir = paths["tools_dir"]

    batch_size = video_settings["batch_size"]
    batch_size_gpu = video_settings["batch_size_gpu"]
    target_fps = video_settings["target_fps"]
    codec_salida = video_settings["codec_salida"]
    name = video_settings["name"]

    time_baseline = bechmark_settings["time_baseline"]
    baseline_method = bechmark_settings["baseline_method"]

    pipeline = config["profiles"][args.profiles]

    save_baseline = args.save_baseline
    save_estimate_baseline = args.save_estimate_baseline

    return input_dir, output_dir, temp_audio_dir, tools_dir, temp_video_mute_dir, temp_frames_origin_dir, temp_frames_filtered_dir, resultados_dir, batch_size, batch_size_gpu, target_fps, codec_salida, name, time_baseline, baseline_method, pipeline, save_baseline, save_estimate_baseline