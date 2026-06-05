# Extrae el MP3/AAC y lo une al video final, lo hace usando la carpeta temp
import subprocess
import os

def extraer_audio(config):
    """
    Funcionalidad Aislada 1: Extracción de Audio.
    Toma el video original y extrae su pista de audio a la carpeta temporal.
    Si el video no tiene audio, maneja el error silenciosamente.
    """
    input_path = os.path.join(config["paths"]["input_dir"], config["video_settings"]["name"])
    audio_dir = config["paths"]["temp_audio_dir"]
    
    # Asegurar que el directorio de audio exista
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        
    audio_output_temp = os.path.join(audio_dir, "audio_original.aac")
    
    print(f"🎵 [Audio] Intentando extraer pista de audio de '{config['video_settings']['name']}'...")
    
    # Comando FFmpeg: -vn (ignorar video), -c:a aac (forzar codec AAC para máxima compatibilidad)
    comando = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-c:a", "aac", audio_output_temp
    ]
    
    try:
        # subprocess.DEVNULL oculta el texto masivo que escupe FFmpeg en la consola
        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ [Audio] Pista de audio extraída y guardada en temporales.")
    except subprocess.CalledProcessError:
        print("⚠️ [Audio] FFmpeg no detectó pista de audio o falló la extracción. El video será mudo.")
        # Limpiar cualquier archivo corrupto que haya intentado crear FFmpeg
        if os.path.exists(audio_output_temp):
            os.remove(audio_output_temp)

def fusionar_audio(config):
    """
    Funcionalidad Aislada 2: Ensamblado (Muxing).
    Toma el video procesado mudo y le inyecta el audio temporal.
    Si no encuentra audio temporal, asume que el video era mudo y solo lo renombra.
    """
    audio_temp = os.path.join(config["paths"]["temp_audio_dir"], "audio_original.aac")
    video_mudo = os.path.join(config["paths"]["output_dir"], "video_procesado_sin_audio.mp4")
    
    nombre_original = config["video_settings"]["name"]
    video_final = os.path.join(config["paths"]["output_dir"], f"Procesado_{nombre_original}")

    # Escenario A: No hay audio (El video original era mudo o falló la extracción)
    if not os.path.exists(audio_temp):
        print("⚠️ [Audio] No se encontró pista de audio. Guardando el resultado como video mudo...")
        if os.path.exists(video_mudo):
            os.rename(video_mudo, video_final)
            print(f"✅ [Final] Video definitivo guardado en: {video_final}")
        return

    # Escenario B: Hay audio, procedemos a fusionar
    print("🎵 [Audio] Inyectando pista de audio al video procesado...")
    
    # Comando FFmpeg: copiar video (-c:v copy) y copiar audio (-c:a copy) sin recodificar
    comando = [
        "ffmpeg", "-y", "-i", video_mudo, "-i", audio_temp,
        "-c:v", "copy", "-c:a", "copy", video_final
    ]
    
    try:
        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Operación de limpieza: borrar el archivo mudo intermedio para no saturar el disco
        if os.path.exists(video_mudo):
            os.remove(video_mudo)
            
        print(f"✅ [Final] Fusión de audio completa. Video definitivo en: {video_final}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ [Audio Crítico] Falló la reincorporación del audio con FFmpeg.")
        # Mecanismo de rescate: al menos dejamos el video mudo con el nombre correcto
        if os.path.exists(video_mudo):
            os.rename(video_mudo, video_final)
            print(f"⚠️ [Rescate] El video mudo se salvó en: {video_final}")