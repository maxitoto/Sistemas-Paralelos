import torch
import torch.nn.functional as F
from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar
from tools.shared.mathstorch import aberracionCromatica, oleo
from tqdm import tqdm
from tools.shared.const.parameters import radioTorch_cpu

class Pipeline(IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar):

    device = torch.device("cpu")

    def __init__(self, config):
        self.filtro_elegido = config["video_settings"].get("filter_type", "oleo")

    def calentar(self):
        pass

    def host_to_device(self, lote_host):
        is_contable = False 
        
        # IMPORTANTE: En CPU mantenemos .float() (32 bits) porque 
        # los procesadores tradicionales suelen dar error con .half() en F.unfold
        lote_device = torch.from_numpy(lote_host).float().permute(0, 3, 1, 2)

        return lote_device, is_contable

    def procesarComputo1(self, lote_device):
        is_contable = True
        B, C, H, W = lote_device.shape
        
        lote_salida = torch.empty_like(lote_device)

        # ==========================================================
        # ESCUDO DE MEMORIA: Apaga el historial de gradientes
        # ==========================================================
        with torch.no_grad():
            for b in tqdm(range(B), desc="Procesando PyTorch CPU", leave=False):
                
                frame_actual = lote_device[b : b + 1]
                #frame_procesado = aberracionCromatica(frame_actual, radioTorch_cpu)
                frame_procesado = oleo(frame_actual, radioTorch_cpu)
                lote_salida[b : b + 1] = frame_procesado

        return lote_salida, is_contable

    def procesarComputo2(self, lote_device):
        is_contable = False 
        return lote_device, is_contable 

    def device_to_host(self, lote_device):
        is_contable = False
        
        lote_host = lote_device.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy()
        
        return lote_host, is_contable
    
    def auxiliar(self):
        pass