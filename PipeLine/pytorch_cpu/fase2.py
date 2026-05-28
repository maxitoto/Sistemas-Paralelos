import torch
import torch.nn.functional as F

def procesar(datos, config):
    tensor_gris_device, device = datos

    # Kernel de Sobel en la memoria principal
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

    # Derivadas parciales (se ejecutan usando múltiples hilos de la CPU)
    gx = F.conv2d(tensor_gris_device, kernel_x, padding=1)
    gy = F.conv2d(tensor_gris_device, kernel_y, padding=1)
    
    # Magnitud y límite vectorial
    tensor_sobel_device = torch.sqrt(gx * gx + gy * gy).clamp(0.0, 255.0)

    return (tensor_gris_device, tensor_sobel_device)