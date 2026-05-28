import torch

def procesar(datos, config):
    tensor_rgb_device, device = datos

    # Separamos los canales Rojo [0], Verde [1] y Azul [2]
    r = tensor_rgb_device[:, 0:1, :, :]
    g = tensor_rgb_device[:, 1:2, :, :]
    b = tensor_rgb_device[:, 2:3, :, :]
    
    # es síncrono por defecto en CPU
    tensor_gris_device = 0.299 * r + 0.587 * g + 0.114 * b
    
    return (tensor_gris_device, device)