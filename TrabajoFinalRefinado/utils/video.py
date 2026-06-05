import cv2
import os
import glob

def abrir_video(config):
    """
    Funcionalidad Aislada 1: Apertura y Metadatos (Carga Perezosa).
    Lee las propiedades del video sin extraer nada a disco.
    """
    # Se actualizó 'nombre_video_entrada' por 'name' según el nuevo config.json
    input_path = os.path.join(config["paths"]["input_dir"], config["video_settings"]["name"])
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ [Error] El archivo de video no existe: {input_path}")
        
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ [Error] OpenCV no pudo decodificar el video: {input_path}")
        
    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    print(f"📼 [Video] Abierto '{config['video_settings']['name']}' | Res: {metadata['width']}x{metadata['height']} | {metadata['fps']} FPS | {metadata['total_frames']} frames")
    
    return cap, metadata

def extraer_frames(config, cap, total_frames):
    """
    Funcionalidad Aislada 2: Extracción a Disco (ORIGEN).
    Extrae los frames físicos EXCLUSIVAMENTE en la carpeta 'origin'.
    """
    # Ahora apunta a temp_frames_origin_dir
    frames_origin_path = config["paths"]["temp_frames_origin_dir"]
    
    if not os.path.exists(frames_origin_path):
        os.makedirs(frames_origin_path)

    print(f"🎞️ [Video] Extrayendo {total_frames} frames a '{frames_origin_path}'...")
    
    frames_extraidos = 0
    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ [Video] Advertencia: Lectura interrumpida en el frame {frame_count}.")
            break
            
        frame_name = os.path.join(frames_origin_path, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_name, frame)
        frames_extraidos += 1
        
    cap.release()
    print(f"✅ [Video] {frames_extraidos} frames originales listos para procesamiento.")
    
    return frames_extraidos

def ensamblar_video(config):
    """
    Funcionalidad Aislada 3: Ensamblado (FILTRADO).
    Lee los frames de la carpeta 'filtered' y ensambla el video mudo.
    """
    # Ahora lee de temp_frames_filtered_dir
    frames_filtered_path = config["paths"]["temp_frames_filtered_dir"]
    output_path = os.path.join(config["paths"]["output_dir"], "video_procesado_sin_audio.mp4")
    
    fps = config["video_settings"]["target_fps"]
    codec = config["video_settings"]["codec_salida"]

    print(f"🎬 [Video] Ensamblando video mudo desde '{frames_filtered_path}'...")
    
    search_pattern = os.path.join(frames_filtered_path, "*.jpg")
    frames_paths = sorted(glob.glob(search_pattern))
    
    if not frames_paths:
        raise FileNotFoundError(f"❌ [Falla Crítica] No se encontraron frames en {frames_filtered_path}. ¿Falló el procesamiento?")
        
    primer_frame = cv2.imread(frames_paths[0])
    height, width = primer_frame.shape[:2]
    es_color = True if len(primer_frame.shape) == 3 else False
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=es_color)
    
    for frame_path in frames_paths:
        frame = cv2.imread(frame_path)
        
        if not es_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        out.write(frame)
        
    out.release()
    print(f"✅ [Video] Ensamblado finalizado.")