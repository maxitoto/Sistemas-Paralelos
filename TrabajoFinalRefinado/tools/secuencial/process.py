# tools/secuencial/process.py
import numpy as np
from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3TransferenciaOut, IFase4Auxiliar
from tools.shared import maths

# Múltiple herencia: Esta clase cumple con TODO el contrato HPC
class Pipeline(IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3TransferenciaOut, IFase4Auxiliar):
    
    def calentar(self):
        pass # Secuencial no necesita calentar
        
    def host_to_device(self, lote_host):
        return lote_host # RAM a RAM (no hace nada)
        
    def procesar(self, lote_device):
        lote_salida = []
        for frame in lote_device:
            frame_gris = maths.filterGris(frame)
            lote_salida.append(frame_gris)
        return np.array(lote_salida, dtype=np.float32)
        
    def device_to_host(self, lote_procesado_device):
        return lote_procesado_device # RAM a RAM (no hace nada)
    
    def auxiliar(self, lote_procesado_device):
        pass #no hace nada en este caso