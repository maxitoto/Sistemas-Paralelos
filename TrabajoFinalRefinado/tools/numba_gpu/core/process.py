import math
import numpy as np
from numba import cuda
from tools.shared.mathsba import oleo
from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar
from tqdm import tqdm

class BasePipeline(IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar):

    def __init__(self, config):
        self.filtro_elegido = config["video_settings"].get("filter_type", "oleo")
        
    def calentar(self):
        pass

    def host_to_device(self, lote_host):
        is_contable = True 
        lote_device = cuda.to_device(lote_host)
        return lote_device, is_contable

    def procesarComputo1(self, lote_device):
        is_contable = True
        
        B, alto, ancho, canales = lote_device.shape
        
        # El molde de salida ya se crea directamente en 1 Byte (uint8) para ahorrar VRAM
        lote_salida_device = cuda.device_array((B, alto, ancho, canales), dtype=np.uint8)
        
        # Adaptamos la configuración de los perfiles (que eran 3D) a un formato 2D
        tpb_x, tpb_y, _ = self.threadsperblock
        threads_2d = (tpb_x, tpb_y)
        
        blocks_x = math.ceil(ancho / tpb_x)
        blocks_y = math.ceil(alto / tpb_y)
        blocks_2d = (blocks_x, blocks_y)
        
        # Iteramos frame a frame
        for b in tqdm(range(B), desc="Procesando Numba GPU", leave=False):
            
            # Extraemos el frame específico (pierde la dimensión B, queda [Alto, Ancho, Canales])
            frame_actual = lote_device[b]
            frame_salida = lote_salida_device[b]
            
            # Lanzamos el kernel 2D para este frame individual
            oleo[blocks_2d, threads_2d](frame_actual, frame_salida)
        
        # Barrera de sincronización obligatoria
        cuda.synchronize()
        
        return lote_salida_device, is_contable

    def procesarComputo2(self, lote_device):
        is_contable = True
        return lote_device, is_contable

    def device_to_host(self, lote_procesado_device):
        is_contable = True 
        lote_host = lote_procesado_device.copy_to_host()
        return lote_host, is_contable
        
    def auxiliar(self):
        pass