from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar
from tools.shared.mathstorch import oleo

import torch
import torch.nn.functional as F

import json

class Pipeline(IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar):

    device = torch.device("cuda")

    def __del__(self):
        torch.cuda.empty_cache()

    def __init__(self, config):
        self.filtro_elegido = config["video_settings"].get("filter_type", "oleo")
        self.batch_size_gpu = config["video_settings"].get("batch_size_gpu", 32)
        pass

    def calentar(self):
        dummy_tensor = torch.zeros((1, 3, 10, 10), dtype=torch.float32, device=self.device)
            
        # Lo procesamos y descartamos el resultado
        _ = oleo(dummy_tensor)
            
        # Sincronizamos para asegurar que el motor arrancó del todo
        torch.cuda.synchronize()

    def host_to_device(self, lote_host):
        is_contable = True 
        
        lote_device = torch.from_numpy(lote_host).float().permute(0, 3, 1, 2).to(self.device)#OpenCv entrega en BGR y PyTorch espera RGB, tengo que permutar los colores (no lo sabia antes de empezar)

        # Por ahora enviamos a la vram la imagen sin procesar

        return lote_device, is_contable

    def procesarComputo1(self, lote_device):

        is_contable = True
        B, C, H, W = lote_device.shape
        
        lote_salida = torch.empty_like(lote_device)

        chunk_size = self.batch_size_gpu

        for i in range(0, B, chunk_size):
            
            chunk_actual = lote_device[i : i + chunk_size]
            
            chunk_procesado = oleo(chunk_actual)
            
            lote_salida[i : i + chunk_size] = chunk_procesado

        torch.cuda.synchronize()

        return lote_salida, is_contable

    def procesarComputo2(self, lote_device):
        is_contable = False 
        return lote_device, is_contable 

    def device_to_host(self, lote_device):
        is_contable = True
        
        # resumen de las operaciones para el lote que llega
        # 1. permute(0, 2, 3, 1): Devolvemos los canales al final para OpenCV [B, Alto, Ancho, C]
        # 2. clamp(0, 255): Cortamos cualquier decimal loco que haya dado la matemática
        # 3. byte(): Volvemos de Float32 a Uint8 (Enteros de 8 bits)
        # 4. cpu(): Bajamos físicamente la información de la Gráfica a la RAM
        # 5. numpy(): Convertimos el Tensor en un Array clásico
        lote_host = lote_device.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()
        
        return lote_host, is_contable
    
    def auxiliar(self):
        pass