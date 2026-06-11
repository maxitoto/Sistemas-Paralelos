import cv2, os, subprocess, glob, shutil, json,numpy as np
try:
    import ffmpeg
    HAS_FFMPEG_LIB = True
except ImportError:
    HAS_FFMPEG_LIB = False
    print("⚠️ [Aviso] Librería ffmpeg-python no encontrada. Se usará el ejecutable local directamente.")

def open_video(input_dir):
    cap = cv2.VideoCapture(input_dir)
    return cap

def split_video(video, temp_frames_origin_dir):
    
    if not os.path.exists(temp_frames_origin_dir):
        os.makedirs(temp_frames_origin_dir)
        
    frames_en_cache = glob.glob(os.path.join(temp_frames_origin_dir, "*.jpg"))
    cantidad_cache = len(frames_en_cache)

    if cantidad_cache > 0:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            filteredPath = config["paths"]["temp_frames_filtered_dir"]
            
        clear_out(filteredPath)
        print(f"⏩ [Video] Se encontraron {cantidad_cache} frames en cache. Saltando extracción.")
        return cantidad_cache

    print("🎬 [Video] No hay caché. Iniciando extracción de frames...")
    frames_guardados = 0
    
    while True:
        ret, frame = video.read()
        if not ret: 
            break
            
        frame_name = f"frame_{frames_guardados:05d}.jpg"
        frame_path = os.path.join(temp_frames_origin_dir, frame_name)
        
        cv2.imwrite(frame_path, frame)
        frames_guardados += 1

    print(f"✅ [Video] Se extrajeron {frames_guardados} frames totales en {temp_frames_origin_dir}")
    return frames_guardados

def merge_video(temp_frames_filtered_dir, temp_video_mute_dir, code, fps):
    search_pattern = os.path.join(temp_frames_filtered_dir, "*.jpg")
    frames_paths = sorted(glob.glob(search_pattern))

    if not frames_paths:
        print("⚠️ [Video] No se encontraron frames procesados para ensamblar.")
        return

    primer_frame = cv2.imread(frames_paths[0])
    alto, ancho, canales = primer_frame.shape 
    resolucion = (ancho, alto)

    fourcc = cv2.VideoWriter_fourcc(*code)
    out = cv2.VideoWriter(temp_video_mute_dir, fourcc, fps, resolucion)

    print(f"🎬 [Video] Ensamblando video mudo a {resolucion} @ {fps} FPS...")

    for path in frames_paths:
        frame = cv2.imread(path)
        out.write(frame)

    out.release()
    print("✅ [Video] Ensamblado de frames finalizado.")

def generar_lotes(temp_frames_origin_dir, batch_size):

    search_pattern = os.path.join(temp_frames_origin_dir, "*.jpg")
    frames_paths = sorted(glob.glob(search_pattern))
    
    if not frames_paths:
        print("⚠️ [Lotes] No se encontraron frames para armar lotes.")
        return
        
    total_frames = len(frames_paths)
    
    for i in range(0, total_frames, batch_size):
        lote_paths = frames_paths[i : i + batch_size]
        lote_frames = []
        
        for path in lote_paths:
            frame = cv2.imread(path)
            lote_frames.append(frame)
            
        batch_np = np.array(lote_frames, dtype=np.uint8)
        nombres_base = [os.path.basename(p) for p in lote_paths]
        
        hay_lotes_restantes = (i + batch_size) < total_frames

        yield batch_np, nombres_base, hay_lotes_restantes
    
    print("🧹 [Lotes] Sin lotes restantes. Limpiando frames originales...")
    clear_out(temp_frames_origin_dir)

def guardar_lote(temp_frames_filtered_dir, lote_procesado, nombres_base):

    if not os.path.exists(temp_frames_filtered_dir):
        os.makedirs(temp_frames_filtered_dir)
        
    if not isinstance(lote_procesado, np.ndarray):
        lote_procesado = np.asarray(lote_procesado)

    for idx, frame in enumerate(lote_procesado):
        
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        out_path = os.path.join(temp_frames_filtered_dir, nombres_base[idx])
        
        cv2.imwrite(out_path, frame)

def extract_audio(ruta_video, temp_audio_dir):
    """
    Extrae el audio del video. Soporta tanto la librería de Python como el ejecutable.
    """
    if not os.path.exists(temp_audio_dir):
        os.makedirs(temp_audio_dir)
        
    output_audio = os.path.join(temp_audio_dir, "audio_original.aac")
    
    print("🎵 [Audio] Extrayendo pista de audio...")

    if HAS_FFMPEG_LIB:
        try:
            stream = ffmpeg.input(ruta_video)
            stream = ffmpeg.output(stream, output_audio, **{'c:a': 'aac', 'vn': None})
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            print("✅ [Audio] Extraído con la librería ffmpeg-python.")
        except Exception as e:
            print(f"❌ [Audio] Error con la librería: {e}")
            
    else:
        comando = [
            'ffmpeg', '-y', 
            '-i', ruta_video, 
            '-vn',               
            '-c:a', 'aac',       # Forzar codec AAC
            output_audio
        ]
        try:
            subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print("✅ [Audio] Extraído usando el ejecutable independiente.")
        except subprocess.CalledProcessError:
            print("⚠️ [Audio] Falló la extracción con ejecutable (Quizás el video no tiene sonido).")

def merge_audio(archivo_video_mudo, archivo_audio_temp, archivo_video_final):
    import shutil
    import subprocess
    import os
    
    if not os.path.exists(archivo_video_mudo):
        print("❌ [Audio] Error fatal: No existe un video procesado para ensamblar. Abortando.")
        return

    if not os.path.exists(archivo_audio_temp):
        print("⚠️ [Audio] No hay pista de audio. Guardando video final sin sonido.")
        shutil.copy(archivo_video_mudo, archivo_video_final)
        return

    command = [
        "ffmpeg", "-y",
        "-i", archivo_video_mudo,
        "-i", archivo_audio_temp,
        "-c:v", "copy",
        "-c:a", "aac",
        archivo_video_final
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("🎵 [Audio] Audio fusionado con éxito.")
    except FileNotFoundError:
        print("⚠️ [Audio] FFmpeg no encontrado en Windows. Guardando video final mudo.")
        shutil.copy(archivo_video_mudo, archivo_video_final)
    except Exception as e:
        print(f"⚠️ [Audio] Falló la fusión de audio: {e}. Guardando video final mudo.")
        shutil.copy(archivo_video_mudo, archivo_video_final)

def clear_out(temp_dir):
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass

    except Exception as e:
        print(f"Error al limpiar la carpeta temporal: {e}")