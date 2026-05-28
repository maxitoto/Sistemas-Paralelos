import torch

def procesar(datos, config):
    tensor_rgb_gpu, device = datos

    # El tensor tiene forma (1, 3, H, W). 
    # Separamos los canales Rojo [0], Verde [1] y Azul [2]
    r = tensor_rgb_gpu[:, 0:1, :, :]
    g = tensor_rgb_gpu[:, 1:2, :, :]
    b = tensor_rgb_gpu[:, 2:3, :, :]
    
    tensor_gris_gpu = 0.299 * r + 0.587 * g + 0.114 * b
    
    torch.cuda.synchronize()

    return (tensor_gris_gpu, device)