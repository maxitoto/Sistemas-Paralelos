import numpy as np
from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar
from tools.shared.maths import oleo
from tqdm import tqdm

class Pipeline(IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar):
    
    def __init__(self, config):
        self.filtro_elegido = config["video_settings"].get("filter_type")
    
    def calentar(self):
        pass

    def host_to_device(self, lote_host):
        is_contable = False
        lote_device = lote_host 
        
        return lote_device, is_contable
        
    def procesarComputo1(self, lote_device):
        is_contable = True
        
        batch_size, alto, ancho, canales = lote_device.shape
        
        # Optimizamos memoria inicializando directamente en uint8
        lote_salida = np.zeros((batch_size, alto, ancho, canales), dtype=np.uint8)
    
        for b in tqdm(range(batch_size), desc="Procesando Secuencial", leave=False):
            frame = lote_device[b]
            for y in range(alto):
                for x in range(ancho):
                    
                    r_final, g_final, b_final = oleo(frame, y, x)

                    lote_salida[b, y, x, 0] = r_final
                    lote_salida[b, y, x, 1] = g_final
                    lote_salida[b, y, x, 2] = b_final

        return lote_salida, is_contable
            
    def procesarComputo2(self, lote_device):
        is_contable = False 
        return lote_device, is_contable
    
    def device_to_host(self, lote_procesado_device):
        is_contable = False
        lote_host_final = lote_procesado_device
        
        return lote_host_final, is_contable
    
    def auxiliar(self):
        pass