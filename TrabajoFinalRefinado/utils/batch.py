import os
import glob
import cv2
import numpy as np

def procesar_en_lotes(config):
    """
    Funcionalidad Aislada 1: Carga y Empaquetado de Alta Precisión.
    Generador que lee los frames originales, arma un tensor NumPy en float32
    y lo entrega al pipeline para evitar truncamientos prematuros.
    """
    origin_dir = config["paths"]["temp_frames_origin_dir"]
    batch_size = config["video_settings"]["batch_size"]
    max_frames = config["video_settings"]["max_frames"]
    
    search_pattern = os.path.join(origin_dir, "*.jpg")
    frames_paths = sorted(glob.glob(search_pattern))
    
    if not frames_paths:
        raise FileNotFoundError(f"❌ [Lotes] No hay frames originales en {origin_dir}. Extrae el video primero.")
        
    if max_frames > 0:
        frames_paths = frames_paths[:max_frames]
        print(f"⚠️ [Lotes] MODO DEBUG ACTIVO: Limitado a procesar {max_frames} frames.")
        
    total_frames = len(frames_paths)
    
    for i in range(0, total_frames, batch_size):
        lote_paths = frames_paths[i : i + batch_size]
        lote_matrices = []
        
        for path in lote_paths:
            frame = cv2.imread(path)
            # Normalizamos el espacio de color
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lote_matrices.append(frame)

        batch_np = np.array(lote_matrices, dtype=np.float32)
        
        nombres_base = [os.path.basename(p) for p in lote_paths]
        
        yield batch_np, nombres_base

def guardar_lote(config, batch_procesado, nombres_base):
    """
    Funcionalidad Aislada 2: Escritura.
    Recibe el lote procesado (idealmente ya casteado a uint8 por las tools) 
    y guarda cada frame individualmente en la carpeta 'filtered'.
    """
    filtered_dir = config["paths"]["temp_frames_filtered_dir"]
    
    if not os.path.exists(filtered_dir):
        os.makedirs(filtered_dir)
        
    # Deducimos si es color o escala de grises
    es_color = True if len(batch_procesado.shape) == 4 else False
    
    for idx, frame in enumerate(batch_procesado):
        
        # Salvavidas de infraestructura: OpenCV necesita uint8 para guardar JPGs correctamente.
        # Si la tool devolvió el tensor en float, lo forzamos a uint8 aquí para evitar crasheos.
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            print(f"⚠️ [Lotes] Forzando a uint8 el frame {idx} de {len(batch_procesado)}, si se utilizo GPU, este paso debe hacer allí mismo.")

        if es_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
        out_path = os.path.join(filtered_dir, nombres_base[idx])
        cv2.imwrite(out_path, frame)