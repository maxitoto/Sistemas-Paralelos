import torch

def procesar(tensor_rgb, config):
    # El orquestador ya le quitó el t_transfer_in, recibimos el tensor puro
    r = tensor_rgb[:, 0:1, :, :]
    g = tensor_rgb[:, 1:2, :, :]
    b = tensor_rgb[:, 2:3, :, :]

    # Matemática en float32
    tensor_gris = 0.299 * r + 0.587 * g + 0.114 * b

    return tensor_gris