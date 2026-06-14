import torch
import torch.nn.functional as F

# Usamos el radio global, sin trampas para PyTorch
from tools.shared.const.parameters import niveles, desplazamiento

def oleo(chunk_tensor, radio):
    

    # B siempre será 1 (un solo frame mandado desde process.py), 
    # pero F.unfold exige que mantengamos la estructura 4D.
    B, C, H, W = chunk_tensor.shape
    
    k = radio * 2 + 1
    kernel_size = k * k

    # 1. Extracción de vecinos
    windows = F.unfold(chunk_tensor, kernel_size=k, padding=radio)
    windows = windows.view(B, C, kernel_size, H * W)

    # 2. Promedio de intensidad
    I = windows.mean(dim=1) 
    
    # Truco de memoria: .short() (16 bits)
    I_bins = ((I / 255.0) * (niveles - 1)).short() 

    # =================================================================
    # 3. VOTACIÓN OPTIMIZADA (SIN ONE_HOT PARA PROTEGER LA RAM)
    # =================================================================
    
    # torch.mode encuentra la moda directamente (valores, indices)
    winning_bin, _ = torch.mode(I_bins, dim=1) 

    # Máscara de 1 byte
    mask = (I_bins == winning_bin.unsqueeze(1)) 

    # 4. Cálculo final de la pincelada
    mask_exp = mask.unsqueeze(1) 
    
    final_color_sum = (windows * mask_exp).sum(dim=2) 

    max_votes = mask.sum(dim=1).clamp(min=1).unsqueeze(1) 
    
    final_color = final_color_sum / max_votes
    
    return final_color.view(B, C, H, W)


def aberracionCromatica(chunk_tensor):
    
    # Separamos los 3 canales: [Batch, Canales, Alto, Ancho]
    R = chunk_tensor[:, 0:1, :, :]
    G = chunk_tensor[:, 1:2, :, :]
    B = chunk_tensor[:, 2:3, :, :]
    
    # ====================================================================
    # Equivalente vectorial a x_rojo = max(0, x - desplazamiento):
    # Rellenamos la izquierda copiando el borde (replicate) y cortamos la derecha
    # ====================================================================
    R_pad = F.pad(R, (desplazamiento, 0, 0, 0), mode='replicate')
    R_shift = R_pad[:, :, :, :-desplazamiento]
    
    # ====================================================================
    # Equivalente vectorial a x_azul = min(ancho - 1, x + desplazamiento):
    # Rellenamos la derecha copiando el borde y cortamos la izquierda
    # ====================================================================
    B_pad = F.pad(B, (0, desplazamiento, 0, 0), mode='replicate')
    B_shift = B_pad[:, :, :, desplazamiento:]
    
    # Juntamos los 3 canales de nuevo (R_desplazado, Verde_intacto, B_desplazado)
    final_color = torch.cat((R_shift, G, B_shift), dim=1)
    
    return final_color