import torch
import torch.nn.functional as F

def procesar(tensor_gris, config):
    device = tensor_gris.device

    # tensores
    kernel_x = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        dtype=torch.float32, device=device
    )
    kernel_y = torch.tensor(
        [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],
        dtype=torch.float32, device=device
    )

    # Convolución
    gx = F.conv2d(tensor_gris, kernel_x, padding=1)
    gy = F.conv2d(tensor_gris, kernel_y, padding=1)
    
    # Raíz cuadrada para la magnitud
    sobel = torch.sqrt(gx**2 + gy**2)

    # magnitud
    tensor_gris_uint8 = tensor_gris.clamp(0, 255).to(torch.uint8)
    gx_uint8 = torch.abs(gx).clamp(0, 255).to(torch.uint8)
    gy_uint8 = torch.abs(gy).clamp(0, 255).to(torch.uint8)
    sobel_uint8 = sobel.clamp(0, 255).to(torch.uint8)

    return (tensor_gris_uint8, gx_uint8, gy_uint8, sobel_uint8)