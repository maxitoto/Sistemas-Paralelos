import torch
import torch.nn.functional as F

def procesar(tensor_gris, config):
    device = tensor_gris.device

    kernel_x = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        dtype=torch.float32, device=device
    )
    kernel_y = torch.tensor(
        [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],
        dtype=torch.float32, device=device
    )

    gx = F.conv2d(tensor_gris, kernel_x, padding=1)
    gy = F.conv2d(tensor_gris, kernel_y, padding=1)
    sobel = torch.sqrt(gx**2 + gy**2)

    # Nivelación de bordes
    sobel[:, :, 0, :] = 0     
    sobel[:, :, -1, :] = 0    
    sobel[:, :, :, 0] = 0     
    sobel[:, :, :, -1] = 0    

    # Casteo a uint8 al finalizar
    tensor_gris_uint8 = torch.round(tensor_gris).clamp(0, 255).to(torch.uint8)
    gx_uint8 = torch.round(torch.abs(gx)).clamp(0, 255).to(torch.uint8)
    gy_uint8 = torch.round(torch.abs(gy)).clamp(0, 255).to(torch.uint8)
    sobel_uint8 = torch.round(sobel).clamp(0, 255).to(torch.uint8)

    return (tensor_gris_uint8, gx_uint8, gy_uint8, sobel_uint8)