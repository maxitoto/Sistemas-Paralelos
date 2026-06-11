import torch
import torch.nn.functional as F

def oleo(chunk_tensor):
    radio=2
    niveles=20
    B, C, H, W = chunk_tensor.shape
    
    k = radio * 2 + 1
    kernel_size = k * k

    # 1. extraccion de vecinos para los N frames al mismo tiempo
    # F.unfold devuelve: (B, C * 25, pixeles)
    windows = F.unfold(chunk_tensor, kernel_size=k, padding=radio)
    
    # reorganizamos para separar la dimensión: (Batch, Canales, 25 vecinos, pixeles), esto lo hace al paloo
    windows = windows.view(B, C, kernel_size, H * W)

    # 2. promedio de intensidad (Colapsamos la dimensión C, que ahora es la dim=1), todo al mismo tiempo
    I = windows.mean(dim=1) # (B, 25, pixeles)
    I_bins = ((I / 255.0) * (niveles - 1)).long()

    # 3. votacion de la moda para N frames, igual rapidizado
    one_hot = F.one_hot(I_bins, num_classes=niveles).float() # (B, 25, pixeles, Niveles)
    one_hot = one_hot.permute(0, 3, 1, 2) # formato: (B, niveles, 25, pixeles)
    
    votes = one_hot.sum(dim=2) # sumamos los votos vecinales: (B, Niveles, pixeles)
    winning_bin = votes.argmax(dim=1) # buscamos la moda: (B, pixeles)

    # 4. calculo final de la pincelada
    # multiplicamos Colores (c) por Votos (v) para cada Frame (b)
    colors_per_bin = torch.einsum('bcnh, bvnh -> bcvh', windows, one_hot) 
    
    # rescatamos los colores especificos del nivel ganador
    winning_bin_exp = winning_bin.unsqueeze(1).unsqueeze(2).expand(B, C, 1, H * W)
    final_color_sum = colors_per_bin.gather(dim=2, index=winning_bin_exp).squeeze(2) 

    # rescatamos la cantidad de votos para promediar
    max_votes = votes.gather(dim=1, index=winning_bin.unsqueeze(1)).clamp(min=1)
    
    # dividimos y aprovechamos el "broadcasting" de pytorch
    final_color = final_color_sum / max_votes
    
    # devolvemos el chunk terminado con su forma original
    return final_color.view(B, C, H, W)