import torch
import torch.nn.functional as F

# Usamos el radio global, sin trampas para PyTorch
from tools.shared.const.parameters import niveles, radio

def oleo(chunk_tensor):

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