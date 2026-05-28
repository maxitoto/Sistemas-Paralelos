import torch
import torch.nn.functional as F

def procesar(datos, config):
    d_img_gris, d_kernel_x, d_kernel_y, device = datos

    # Convolución bidimensional hiperoptimizada usando cuDNN
    gx = F.conv2d(d_img_gris, d_kernel_x, padding=1)
    gy = F.conv2d(d_img_gris, d_kernel_y, padding=1)
    
    # Teorema de Pitágoras y recorte (clamp)
    d_img_sobel = torch.sqrt(gx * gx + gy * gy).clamp(0.0, 255.0)
    
    # Sincronizamos para que el orquestador mida el cómputo exacto
    torch.cuda.synchronize()

    return (d_img_gris, d_img_sobel)