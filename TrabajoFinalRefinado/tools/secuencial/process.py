from hmac import new

import numpy as np
from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar
from tools.shared.maths import oleo

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
        
        lote_salida = np.zeros((batch_size, alto, ancho, canales), dtype=np.float32)
    
        radio = 2 

        for b in range(batch_size):
            for y in range(alto):
                for x in range(ancho):
                    
                    y_min = max(0, y - radio)
                    y_max = min(alto, y + radio + 1)
                    x_min = max(0, x - radio)
                    x_max = min(ancho, x + radio + 1)
                    
                    kernel = lote_device[b, y_min:y_max, x_min:x_max]
                    
                    r_final, g_final, b_final = oleo(kernel)

                    lote_salida[b, y, x, 0] = r_final
                    lote_salida[b, y, x, 1] = g_final
                    lote_salida[b, y, x, 2] = b_final

        return lote_salida.astype(np.uint8), is_contable
            

    def procesarComputo2(self, lote_device):
        is_contable = False 
        return lote_device, is_contable
    
    def device_to_host(self, lote_procesado_device):
        is_contable = False
        lote_host_final = lote_procesado_device
        
        return lote_host_final, is_contable
    
    def auxiliar(self):
        pass