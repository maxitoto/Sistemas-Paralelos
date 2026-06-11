import math
from numba import cuda, int32, float32, uint8

@cuda.jit
def oleo(lote_in, lote_out):
    niveles = 20
    radio = 2

    # Leemos la forma directamente del tensor principal
    batch_size = lote_in.shape[0]
    alto = lote_in.shape[1]
    ancho = lote_in.shape[2]

    # 1. Pedimos las coordenadas exactas de este hilo
    x, y, b = cuda.grid(3) # ahora usamos tres dimensiones en vez de dos, frames por pixel por pixel. mismo pixel modificado en todos los frames al mismo tiempo
    
    # 2. Escudo de seguridad
    if x < ancho and y < alto and b < batch_size:
        
        # 3. RESERVA DE MEMORIA ESTÁTICA PRIVADA
        conteos = cuda.local.array(niveles, dtype=int32)
        suma_r = cuda.local.array(niveles, dtype=float32)
        suma_g = cuda.local.array(niveles, dtype=float32)
        suma_b = cuda.local.array(niveles, dtype=float32)
        
        for i in range(niveles):
            conteos[i] = 0
            suma_r[i] = 0.0
            suma_g[i] = 0.0
            suma_b[i] = 0.0
            
        # 4. Clipping de bordes
        y_min = max(0, y - radio)
        y_max = min(alto, y + radio + 1)
        x_min = max(0, x - radio)
        x_max = min(ancho, x + radio + 1)
        
        # 5. Evaluación Vecinal
        for vy in range(y_min, y_max):
            for vx in range(x_min, x_max):
                
                rf = float32(lote_in[b, vy, vx, 0])
                gf = float32(lote_in[b, vy, vx, 1])
                bf = float32(lote_in[b, vy, vx, 2])
                
                intensidad = int( (((rf + gf + bf) / 3.0) / 255.0) * (niveles - 1) )
                
                conteos[intensidad] += 1
                suma_r[intensidad] += rf 
                suma_g[intensidad] += gf
                suma_b[intensidad] += bf
                
        # 6. Moda
        max_votos = 0
        nivel_ganador = 0
        for i in range(niveles):
            if conteos[i] > max_votos:
                max_votos = conteos[i]
                nivel_ganador = i
                
        # 7. Pincelada Final
        if max_votos == 0:
            lote_out[b, y, x, 0] = 0
            lote_out[b, y, x, 1] = 0
            lote_out[b, y, x, 2] = 0
        else:
            lote_out[b, y, x, 0] = uint8(suma_r[nivel_ganador] / max_votos)
            lote_out[b, y, x, 1] = uint8(suma_g[nivel_ganador] / max_votos)
            lote_out[b, y, x, 2] = uint8(suma_b[nivel_ganador] / max_votos)