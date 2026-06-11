from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar
from tools.shared.mathstorch import oleo

import torch
import torch.nn.functional as F

from tqdm import tqdm

class Pipeline(IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar):

    device = torch.device("cuda")

    def __del__(self):
        torch.cuda.empty_cache()

    def __init__(self, config):
        self.filtro_elegido = config["video_settings"].get("filter_type", "oleo")

    def calentar(self):
        # Usamos B=1 explícitamente para el calentamiento
        dummy_tensor = torch.zeros((1, 3, 10, 10), dtype=torch.float16, device=self.device)
        _ = oleo(dummy_tensor)
        torch.cuda.synchronize()

    def host_to_device(self, lote_host):
        is_contable = True 
        
        # Pasamos a Media Precisión (16 bits) para ahorrar VRAM
        lote_device = torch.from_numpy(lote_host).half().permute(0, 3, 1, 2).to(self.device)

        return lote_device, is_contable

    def procesarComputo1(self, lote_device):
        is_contable = True
        B, C, H, W = lote_device.shape
        
        lote_salida = torch.empty_like(lote_device)

        # Iteramos frame por frame para evitar el colapso de F.unfold
        for b in tqdm(range(B), desc="Procesando PyTorch GPU", leave=False):
            
            frame_actual = lote_device[b : b + 1]
            frame_procesado = oleo(frame_actual)
            lote_salida[b : b + 1] = frame_procesado

        torch.cuda.synchronize()

        return lote_salida, is_contable

    def procesarComputo2(self, lote_device):
        is_contable = False 
        return lote_device, is_contable 

    def device_to_host(self, lote_device):
        is_contable = True
        
        lote_host = lote_device.permute(0, 2, 3, 1).clamp(0, 255).byte().cpu().numpy()
        
        return lote_host, is_contable
    
    def auxiliar(self):
        pass