import torch
import torch.nn.functional as F

def procesar(tensor_gris, config):
    device = tensor_gris.device

    # Kernels de Sobel en tensores!
    kernel_x = torch.tensor(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        dtype=torch.float32, device=device
    )
    kernel_y = torch.tensor(
        [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]],
        dtype=torch.float32, device=device
    )

    # Convolución optimizada con múltiples hilos de la CPU
    gx = F.conv2d(tensor_gris, kernel_x, padding=1)
    gy = F.conv2d(tensor_gris, kernel_y, padding=1)
    
    # magnitud
    sobel = torch.sqrt(gx**2 + gy**2)

    # asignacion, magnitud y casteo a uint8
    # Usamos .clamp(0, 255) equivalente al np.clip de tu código
    tensor_gris_uint8 = tensor_gris.clamp(0, 255).to(torch.uint8)
    gx_uint8 = torch.abs(gx).clamp(0, 255).to(torch.uint8)
    gy_uint8 = torch.abs(gy).clamp(0, 255).to(torch.uint8)
    sobel_uint8 = sobel.clamp(0, 255).to(torch.uint8)

    # Devolvemos los 4 tensores tal como lo hace tu NumPy
    return (tensor_gris_uint8, gx_uint8, gy_uint8, sobel_uint8)