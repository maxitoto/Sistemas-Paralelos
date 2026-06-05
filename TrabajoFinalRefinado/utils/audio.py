import subprocess
import os

def extraer_audio(config):
    input_path = os.path.join(config["paths"]["input_dir"], config["video_settings"]["name"])
    audio_dir = config["paths"]["temp_audio_dir"]
    
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        
    audio_output_temp = os.path.join(audio_dir, "audio_original.aac")
    
    # NUEVO: Ruta absoluta a tu ejecutable local
    ruta_ffmpeg = os.path.abspath("ffmpeg")
    
    print(f"🎵 [Audio] Intentando extraer pista de audio de '{config['video_settings']['name']}'...")
    
    # Reemplazamos "ffmpeg" por ruta_ffmpeg
    comando = [
        ruta_ffmpeg, "-y", "-i", input_path,
        "-vn", "-c:a", "aac", audio_output_temp
    ]
    
    try:
        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ [Audio] Pista de audio extraída y guardada.")
    except subprocess.CalledProcessError:
        print("⚠️ [Audio] FFmpeg no detectó pista o falló. El video será mudo.")
        if os.path.exists(audio_output_temp):
            os.remove(audio_output_temp)

def fusionar_audio(config):
    audio_temp = os.path.join(config["paths"]["temp_audio_dir"], "audio_original.aac")
    video_mudo = os.path.join(config["paths"]["output_dir"], "video_procesado_sin_audio.mp4")
    
    nombre_original = config["video_settings"]["name"]
    video_final = os.path.join(config["paths"]["output_dir"], f"Procesado_{nombre_original}")

    if not os.path.exists(audio_temp):
        print("⚠️ [Audio] No se encontró pista de audio. Guardando como video mudo...")
        if os.path.exists(video_mudo):
            os.rename(video_mudo, video_final)
            print(f"✅ [Final] Video guardado en: {video_final}")
        return

    print("🎵 [Audio] Inyectando pista de audio...")
    
    # NUEVO: Ruta absoluta a tu ejecutable local
    ruta_ffmpeg = os.path.abspath("ffmpeg")
    
    # Reemplazamos "ffmpeg" por ruta_ffmpeg
    comando = [
        ruta_ffmpeg, "-y", "-i", video_mudo, "-i", audio_temp,
        "-c:v", "copy", "-c:a", "copy", video_final
    ]
    
    try:
        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(video_mudo):
            os.remove(video_mudo)
        print(f"✅ [Final] Fusión completa. Video definitivo en: {video_final}")
    except subprocess.CalledProcessError:
        print(f"❌ [Audio Crítico] Falló la reincorporación del audio.")
        if os.path.exists(video_mudo):
            os.rename(video_mudo, video_final)
            print(f"⚠️ [Rescate] El video mudo se salvó en: {video_final}")