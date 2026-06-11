import math
import numpy as np
from numba import cuda
from tools.shared.mathsba import oleo
from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar

class BasePipeline(IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar):

    def __init__(self, config):
        self.filtro_elegido = config["video_settings"].get("filter_type", "oleo")
        self.batch_size_gpu = config["video_settings"].get("batch_size_gpu", 32)
        
    def calentar(self):
        pass

    def host_to_device(self, lote_host):
        is_contable = True 
        lote_device = cuda.to_device(lote_host)
        return lote_device, is_contable


    '''
    ¿Por qué PyTorch explotaba y Numba (probablemente) no?
    En PyTorch, tuvimos que usar la función mágica F.unfold(). Esa función tiene un defecto grave: para aislar a los 25 vecinos, duplica la memoria de la imagen 25 veces. Si le mandabas 64 frames, en la VRAM se expandían como si fueran ¡1,600 frames! Por eso la memoria colapsaba al instante.

    En Numba, nosotros escribimos el código a bajo nivel. Nuestro hilo no duplica la imagen entera; simplemente lee un píxel a la vez usando las coordenadas de memoria. Si le mandas 64 frames, en la VRAM solo ocupan 64 frames. Es unas 25 veces más eficiente en memoria que PyTorch, por lo que probablemente tu GPU soporte enviar los 64 de un solo golpe sin sudar.
    '''
    def procesarComputo1(self, lote_device):
        is_contable = True
        
        B, alto, ancho, canales = lote_device.shape
        
        lote_salida_device = cuda.device_array((B, alto, ancho, canales), dtype=np.uint8)
        
        chunk_size = self.batch_size_gpu
        
        for i in range(0, B, chunk_size):
            
            chunk_actual = lote_device[i : i + chunk_size]
            b_chunk = chunk_actual.shape[0]
            
            # Recalculamos la Malla (Grid) basada SOLO en este pedacito
            blocks_x = math.ceil(ancho / self.threadsperblock[0])
            blocks_y = math.ceil(alto / self.threadsperblock[1])
            blocks_z = math.ceil(b_chunk / self.threadsperblock[2]) # Usamos b_chunk
            blockspergrid = (blocks_x, blocks_y, blocks_z)
            
            # Lanzamos el kernel exclusivamente para este pedazo
            # Y guardamos el resultado directamente en la ranura correspondiente del molde final
            oleo[blockspergrid, self.threadsperblock](chunk_actual, lote_salida_device[i : i + chunk_size])
        
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