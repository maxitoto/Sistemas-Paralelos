import time
import platform
import json

class GestorEstadisticas:
    """
    Recolector centralizado de métricas. Mide tiempos, evalúa el estado del Baseline
    (Crear, Reemplazar o Reutilizar) y persiste los datos en config.json.
    """
    def __init__(self, config, metadata_video):
        self.config = config
        self.metadata = metadata_video
        
        self.cronometros = {}
        self.resultados = []
        
        # --- 1. LECTURA DEL ESTADO INICIAL ---
        bench_settings = config.get("benchmark_settings", {})
        self.baseline_tool = bench_settings.get("baseline_tool", "secuencial")
        self.save_baseline = bench_settings.get("save_baseline", True)
        
        # Extraemos el tiempo si existe (puede ser un número o None si nunca se ejecutó)
        self.baseline_computo = bench_settings.get("time_baseline", None)
        
        if self.baseline_computo is not None:
            if self.save_baseline:
                print(f"📊 [Estadísticas] Baseline de {self.baseline_computo}s encontrado, pero será REEMPLAZADO en esta ejecución.")
            else:
                print(f"📊 [Estadísticas] Baseline de {self.baseline_computo}s encontrado. Será REUTILIZADO (Modo solo lectura).")
        else:
            print(f"📊 [Estadísticas] No existe un Baseline. A la espera de '{self.baseline_tool}' para calcular uno nuevo.")

    # ==========================================
    # API DE CRONOMETRAJE
    # ==========================================
    def tic(self, fase):
        self._sincronizar_gpu()
        self.cronometros[fase] = time.perf_counter()

    def toc(self, fase):
        self._sincronizar_gpu()
        inicio = self.cronometros.pop(fase, None)
        if inicio is None:
            raise RuntimeError(f"❌ Cronómetro '{fase}' no fue iniciado.")
        return time.perf_counter() - inicio

    def _sincronizar_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    # ==========================================
    # MÁQUINA DE ESTADOS DEL BASELINE
    # ==========================================
    def registrar_corrida(self, herramienta, t_lectura, t_computo, t_escritura, memoria_stats):
        t_total_pipeline = t_lectura + t_computo + t_escritura
        total_frames = self.metadata["total_frames"]
        fps_efectivos = total_frames / t_total_pipeline if t_total_pipeline > 0 else 0

        # CASO A: Se está ejecutando la herramienta designada como Baseline
        if herramienta == self.baseline_tool:
            
            if self.save_baseline:
                # MODO REEMPLAZO: Pisamos el valor en memoria y en el disco
                self.baseline_computo = round(t_computo, 4)
                self.config["benchmark_settings"]["time_baseline"] = self.baseline_computo
                
                try:
                    with open('config.json', 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, indent=4)
                    print(f"💡 [Estadísticas] Baseline REEMPLAZADO en config.json -> Nuevo valor: {self.baseline_computo}s")
                except Exception as e:
                    print(f"⚠️ [Advertencia] No se pudo escribir en config.json. Detalle: {e}")
                
                speedup = 1.0 
            
            else:
                # MODO REUTILIZACIÓN (Solo lectura)
                if self.baseline_computo is not None and self.baseline_computo > 0:
                    # Compara cómo rindió la herramienta base hoy vs el tiempo histórico que está en el JSON
                    speedup = self.baseline_computo / t_computo
                    print(f"💡 [Estadísticas] Reutilizando Baseline histórico. (Rendimiento actual vs histórico: {speedup:.2f}x)")
                else:
                    # Falla de seguridad: save_baseline es false pero borraron el número del JSON
                    self.baseline_computo = round(t_computo, 4)
                    speedup = 1.0
                    print(f"⚠️ [Estadísticas] save_baseline es 'false' pero no existía tiempo previo. Usando {self.baseline_computo}s solo temporalmente.")
        
        # CASO B: Se están ejecutando otras herramientas (Numba / PyTorch)
        else:
            if self.baseline_computo is not None and self.baseline_computo > 0:
                speedup = self.baseline_computo / t_computo
            else:
                print(f"⚠️ [Estadísticas] Evaluando '{herramienta}' sin un Baseline válido. Speed-Up será 0.0x")
                speedup = 0.0

        # Construcción y guardado de la fila de resultados
        fila = {
            "Metodo": herramienta,
            "Frames": total_frames,
            "T_Lectura(s)": round(t_lectura, 4),
            "T_Computo(s)": round(t_computo, 4),
            "T_Escritura(s)": round(t_escritura, 4),
            "T_Total(s)": round(t_total_pipeline, 4),
            "FPS_Efectivos": round(fps_efectivos, 2),
            "RAM_Pico(MB)": memoria_stats.get("RAM_Peak_MB", 0),
            "VRAM_Pico(MB)": memoria_stats.get("VRAM_Peak_MB", 0),
            "Speed-Up": f"{speedup:.2f}x"
        }
        
        self.resultados.append(fila)
        print(f"📊 [{herramienta.upper()}] T_Total: {fila['T_Total(s)']}s | FPS: {fila['FPS_Efectivos']} | Speed-Up: {fila['Speed-Up']}")

    # ==========================================
    # EXPORTACIÓN
    # ==========================================
    def obtener_datos_exportacion(self):
        head = [
            ["CONTEXTO DEL EXPERIMENTO:", f"Video {self.metadata['width']}x{self.metadata['height']} | {self.metadata['fps']} FPS Originales"],
            ["HARDWARE DETALLADO:"],
            ["CPU:", f"OS: {platform.system()} {platform.release()} | Arquitectura: {platform.machine()}"],
            ["GPU:", f"Driver: {self.metadata['driver']} | Version: {self.metadata['version']} | Capacidad: {self.metadata['gpu_memory']} MB"],
            ["BATCH SIZE:", str(self.config['video_settings']['batch_size'])],
            []
        ]
        
        if not self.resultados:
            return head, []

        nombres_columnas = list(self.resultados[0].keys())
        head.append(nombres_columnas)
        cuerpo = [list(res.values()) for res in self.resultados]

        return head, cuerpo