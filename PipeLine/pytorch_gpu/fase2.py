import torch
import torch.nn.functional as F

def procesar(datos, config):
    tensor_gris_gpu, device = datos

    # Kernel de Sobel pero en vram
    kernel_x = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        dtype=torch.float32,
        device=device,
    )
    kernel_y = torch.tensor(
        [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],
        dtype=torch.float32,
        device=device,
    )

    # Derivadas parciales (padding es porque la imagen es cuadrada)
    gx = F.conv2d(tensor_gris_gpu, kernel_x, padding=1)
    gy = F.conv2d(tensor_gris_gpu, kernel_y, padding=1)
    
    # Magnitud y límite vectorial
    tensor_sobel_gpu = torch.sqrt(gx * gx + gy * gy).clamp(0.0, 255.0)
    
    torch.cuda.synchronize()

    return (tensor_gris_gpu, tensor_sobel_gpu)