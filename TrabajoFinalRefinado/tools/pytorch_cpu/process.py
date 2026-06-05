import torch
import torch.nn.functional as F
import numpy as np
from tools.shared.interfaces import IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar

class Pipeline(IFase0WarmUp, IFase1TransferenciaIn, IFase2Computo, IFase3Computo, IFase4TransferenciaOut, IFase5Auxiliar):
    
    def __init__(self, config):
        self.filtro_elegido = config["video_settings"].get("filter_type")
        self.radio = 2       # Ventana de 5x5
        self.niveles = 20    # Tonos de pintura
        
        self.device = torch.device('cpu') 

    def calentar(self):
        """ 
        FASE 0: Inicializamos los motores de PyTorch en CPU 
        para que reserve la memoria necesaria antes de medir el tiempo.
        """
        dummy = torch.zeros((1, 3, 10, 10), dtype=torch.float32, device=self.device)
        _ = F.unfold(dummy, kernel_size=5, padding=2)
         
    def host_to_device(self, lote_host):
        """
        FASE 1: Transferencia de RAM a PyTorch Tensor
        Como seguimos en CPU, no hay viaje físico por cable PCIe. El tiempo NO se cuenta.
        """
        is_contable = False
        
        # PyTorch espera que el color esté en la dimensión 1: (Batch, Canales, Alto, Ancho)
        # NumPy lo entrega como: (Batch, Alto, Ancho, Canales)
        lote_tensor = torch.from_numpy(lote_host).float().permute(0, 3, 1, 2)
        
        return lote_tensor, is_contable
        
    def procesarComputo1(self, lote_device):
        """
        FASE 2: Computo Vectorizado del Filtro Óleo.
        ¡Aquí sí contamos el tiempo! Reemplazamos 150 millones de iteraciones de 
        bucles Python por álgebra tensorial en backend C++.
        """
        is_contable = True
        
        # B = Batch (30), C = Canales (3), H = Alto, W = Ancho
        B, C, H, W = lote_device.shape
        k = self.radio * 2 + 1   # Tamaño del kernel (5)
        kernel_size = k * k      # Píxeles por ventana (25)

        # Matriz vacía para guardar el lote terminado
        lote_salida = torch.empty_like(lote_device)

        # Iteramos el Batch, pero NO los píxeles (eso lo vectoriza PyTorch)
        for b in range(B):
            # Aislamos el frame y le agregamos la dimensión batch para unfold: (1, 3, H, W)
            frame = lote_device[b].unsqueeze(0) 

            # 1. EXTRACCIÓN ESPACIAL: Recortamos los 25 vecinos para todos los píxeles AL MISMO TIEMPO
            # windows shape: (1, Canales * 25, Alto * Ancho)
            windows = F.unfold(frame, kernel_size=k, padding=self.radio)
            
            # Reorganizamos para separar los canales de los vecinos: (3, 25, Píxeles_Totales)
            windows = windows.view(C, kernel_size, H * W)

            # 2. CÁLCULO DE INTENSIDAD (R+G+B / 3)
            # colapsamos la dimensión de canales (dim=0)
            I = windows.mean(dim=0) # Queda: (25 vecinos, Píxeles_Totales)
            I_bins = ((I / 255.0) * (self.niveles - 1)).long()

            # 3. SISTEMA DE VOTACIÓN (One-Hot Encoding)
            # Creamos una matriz donde cada voto es un '1' en su nivel correspondiente
            one_hot = F.one_hot(I_bins, num_classes=self.niveles).float() # (25, Píxeles, 20)
            one_hot = one_hot.permute(2, 0, 1) # (20 niveles, 25 vecinos, Píxeles)

            # Sumamos los votos de los 25 vecinos para cada nivel
            votes = one_hot.sum(dim=1) # (20 niveles, Píxeles)
            
            # Buscamos el nivel con más votos (la Moda)
            winning_bin = votes.argmax(dim=0) # (Píxeles)

            # 4. SUMA DE COLORES DE LA PINCELADA
            # Usamos 'einsum' (Notación de Einstein) para multiplicar los colores por los votos reales
            # c=canales, n=vecinos, l=píxeles, v=niveles (tonos)
            colors_per_bin = torch.einsum('cnl, vnl -> cvl', windows, one_hot) # (3, 20, Píxeles)

            # 5. EXTRACCIÓN DEL COLOR FINAL
            # Expandimos el índice ganador para usar 'gather' y rescatar los colores del nivel ganador
            winning_bin_exp = winning_bin.view(1, 1, H * W).expand(C, 1, H * W)
            final_color_sum = colors_per_bin.gather(dim=1, index=winning_bin_exp).squeeze(1) # (3, Píxeles)

            # Extraemos la cantidad de votos ganadores (clamp evita divisiones por 0)
            max_votes = votes.gather(dim=0, index=winning_bin.unsqueeze(0)).expand(C, H * W).clamp(min=1)

            # Promedio final = Suma de colores / Votos
            final_color = final_color_sum / max_votes
            
            # Devolvemos el array 1D a su forma 2D original (3, Alto, Ancho)
            lote_salida[b] = final_color.view(C, H, W)

        return lote_salida, is_contable
        
    def procesarComputo2(self, lote_device):
        is_contable = False
        return lote_device, is_contable

    def device_to_host(self, lote_procesado_device):
        """
        FASE 4: Transferencia de Vuelta.
        Aseguramos que el Tensor vuelva a ser un Array de NumPy de 8-bits legible por OpenCV.
        """
        is_contable = False
        
        # Volvemos a (Batch, Alto, Ancho, Canales), recortamos a rango seguro (0-255) y convertimos a Entero (Byte)
        lote_host = lote_procesado_device.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy()
        
        return lote_host, is_contable
    
    def auxiliar(self):
        pass