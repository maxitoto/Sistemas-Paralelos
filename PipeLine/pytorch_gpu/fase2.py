import torch
import torch.nn.functional as F

def procesar(datos, config):
    d_img_gris, d_kernel_x, d_kernel_y, device = datos

    # Convolución bidimensional hiperoptimizada usando cuDNN
    gx = F.conv2d(d_img_gris, d_kernel_x, padding=1)
    gy = F.conv2d(d_img_gris, d_kernel_y, padding=1)
    
    # Teorema de Pitágoras y magnitud pura en float32
    d_img_sobel = torch.sqrt(gx * gx + gy * gy)

    # --- NIVELACIÓN DE MARCO (Apagado de bordes) ---
    d_img_sobel[:, :, 0, :] = 0     # Borde superior
    d_img_sobel[:, :, -1, :] = 0    # Borde inferior
    d_img_sobel[:, :, :, 0] = 0     # Borde izquierdo
    d_img_sobel[:, :, :, -1] = 0    # Borde derecho
    
    # --- CASTEO FINAL A UINT8 (Con Redondeo) ---
    # Igualamos a numpy_cpu y numba: redondeo, límite a 255 y compactación a 1 byte
    d_img_gris_uint8 = torch.round(d_img_gris).clamp(0, 255).to(torch.uint8)
    gx_uint8 = torch.round(torch.abs(gx)).clamp(0, 255).to(torch.uint8)
    gy_uint8 = torch.round(torch.abs(gy)).clamp(0, 255).to(torch.uint8)
    d_img_sobel_uint8 = torch.round(d_img_sobel).clamp(0, 255).to(torch.uint8)

    # Sincronizamos para que el orquestador mida el cómputo exacto
    torch.cuda.synchronize()

    # Retornamos las 4 matrices ultraligeras para la Fase 3
    return (d_img_gris_uint8, gx_uint8, gy_uint8, d_img_sobel_uint8)