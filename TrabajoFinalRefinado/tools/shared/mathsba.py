import math
from tools.shared.const.parameters import niveles, radio, desplazamiento
from numba import cuda, int32, float32, uint8

@cuda.jit
def oleo(frame_in, frame_out):

    # Leemos la forma de la imagen individual (Alto, Ancho, Canales)
    alto = frame_in.shape[0]
    ancho = frame_in.shape[1]

    # 1. Pedimos las coordenadas en 2D (Eliminamos la dimensión 'b')
    x, y = cuda.grid(2) 
    
    # 2. Escudo de seguridad
    if x < ancho and y < alto:
        
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
                
                # Extraemos los colores del pixel (ya no usamos índice 'b')
                rf = float32(frame_in[vy, vx, 0])
                gf = float32(frame_in[vy, vx, 1])
                bf = float32(frame_in[vy, vx, 2])
                
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
            frame_out[y, x, 0] = 0
            frame_out[y, x, 1] = 0
            frame_out[y, x, 2] = 0
        else:
            frame_out[y, x, 0] = uint8(suma_r[nivel_ganador] / max_votos)
            frame_out[y, x, 1] = uint8(suma_g[nivel_ganador] / max_votos)
            frame_out[y, x, 2] = uint8(suma_b[nivel_ganador] / max_votos)


@cuda.jit
def aberracionCromatica(frame_in, frame_out):

    alto = frame_in.shape[0]
    ancho = frame_in.shape[1]

    x, y = cuda.grid(2) 
    
    # Escudo de seguridad de la malla
    if x < ancho and y < alto:
        
        x_rojo = max(0, x - desplazamiento)
        x_azul = min(ancho - 1, x + desplazamiento)
        
        # Asignamos directamente los píxeles (Complejidad O(1) - Cero memoria extra)
        frame_out[y, x, 0] = frame_in[y, x_rojo, 0]
        frame_out[y, x, 1] = frame_in[y, x, 1]
        frame_out[y, x, 2] = frame_in[y, x_azul, 2]