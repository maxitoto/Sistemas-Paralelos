import torch

def procesar(datos, config):
    d_img_in, d_kernel_x, d_kernel_y, device = datos

    # Extracción de canales
    r = d_img_in[:, 0:1, :, :]
    g = d_img_in[:, 1:2, :, :]
    b = d_img_in[:, 2:3, :, :]
    
    # Cálculo vectorial puro en VRAM (exactamente igual a tu lógica Numba)
    d_img_gris = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Obligamos a esperar que termine la matemática antes de detener el reloj del orquestador
    torch.cuda.synchronize()

    # Retornamos el resultado y seguimos arrastrando los kernels pre-alojados
    return (d_img_gris, d_kernel_x, d_kernel_y, device)