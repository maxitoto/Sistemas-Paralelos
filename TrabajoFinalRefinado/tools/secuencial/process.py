import numpy as np
from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3TransferenciaOut
from tools.shared import maths

class Pipeline(IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3TransferenciaOut):
    
    def __init__(self, config):
        # Al nacer la herramienta, lee el JSON y guarda qué filtro debe usar
        self.filtro_elegido = config["video_settings"].get("filter_type")
    
    def calentar(self):
        pass
        
    def host_to_device(self, lote_host):
        return lote_host
        
    def procesar(self, lote_device):
        """ FASE 2: Iterador Totalmente Agnóstico """
        batch_size, alto, ancho, canales_in = lote_device.shape
        
        info_filtro = maths.obtener_filtro(self.filtro_elegido)
        filtro_func = info_filtro["func"]
        canales_out = info_filtro["canales_out"]
        
        if canales_out == 1:
            lote_salida = np.zeros((batch_size, alto, ancho), dtype=np.float32)
        else:
            lote_salida = np.zeros((batch_size, alto, ancho, canales_out), dtype=np.float32)

        for b in range(batch_size):
            for y in range(alto):
                for x in range(ancho):
                    
                    pixel = lote_device[b, y, x]
                    
                    valor = filtro_func(pixel[0], pixel[1], pixel[2])

                    lote_salida[b, y, x] = valor

        return np.array(lote_salida, dtype=np.uint8)
        
    def auxiliar(self):
        pass

    def device_to_host(self, lote_procesado_device):
        return lote_procesado_device