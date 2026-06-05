# para registrar el pico de memoria RAM consumida por tu proceso de Python y vRAM consumida por tu proceso de PyTorch
import os
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class MonitorMemoria:
    """
    Herramienta de diagnóstico para rastrear el consumo máximo (Peak) 
    de RAM física y VRAM (memoria de video) durante el benchmarking.
    """
    def __init__(self):
        # Enganchamos el monitor exclusivamente al proceso actual de Python
        self.proceso = psutil.Process(os.getpid())
        self.ram_inicial = 0
        self.ram_pico = 0
        self.vram_pico = 0
        
        # En Mac (Apple Silicon), cuda.is_available() dará False, lo cual es correcto 
        # para evitar que PyTorch busque memoria de Nvidia que no existe.
        self.tiene_cuda = HAS_TORCH and torch.cuda.is_available()

    def iniciar_registro(self):
        """Toma una instantánea de la memoria ANTES de empezar."""
        self.ram_inicial = self.proceso.memory_info().rss
        self.ram_pico = self.ram_inicial
        
        if self.tiene_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        print("📊 [Memoria] Monitor de recursos iniciado.")

    def actualizar_pico_ram(self):
        """Revisa si el consumo actual superó al máximo registrado históricamente."""
        uso_actual = self.proceso.memory_info().rss
        if uso_actual > self.ram_pico:
            self.ram_pico = uso_actual

    def obtener_resultados(self):
        """Calcula la diferencia neta de memoria utilizada y devuelve los picos en MB."""
        consumo_ram_bytes = self.ram_pico - self.ram_inicial
        consumo_ram_mb = max(0, consumo_ram_bytes / (1024 * 1024))
        
        consumo_vram_mb = 0
        if self.tiene_cuda:
            consumo_vram_bytes = torch.cuda.max_memory_allocated()
            consumo_vram_mb = consumo_vram_bytes / (1024 * 1024)

        resultados = {
            "RAM_Peak_MB": round(consumo_ram_mb, 2),
            "VRAM_Peak_MB": round(consumo_vram_mb, 2)
        }
        return resultados