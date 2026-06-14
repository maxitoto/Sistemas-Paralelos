from tools.shared.const.parameters import niveles, radio_sec, desplazamiento

def oleo(frame, y, x):
    """
    info util: https://docs.gimp.org/es/gimp-filter-oilify.html
    """

    radio = radio_sec

    # Dimensiones del frame
    alto, ancho, _ = frame.shape

    # Limites de la ventana 
    y_min = max(0, y - radio)
    y_max = min(alto, y + radio + 1)
    x_min = max(0, x - radio)
    x_max = min(ancho, x + radio + 1)
    
    # kernel seguro
    kernel = frame[y_min:y_max, x_min:x_max]

    conteos = [0] * niveles
    suma_r = [0.0] * niveles
    suma_g = [0.0] * niveles
    suma_b = [0.0] * niveles

    for fila in kernel:
        for r, g, b in fila:
            rf, gf, bf = float(r), float(g), float(b)
            intensidad = int( (((rf + gf + bf) / 3.0) / 255.0) * (niveles - 1) )
            
            conteos[intensidad] += 1
            suma_r[intensidad] += rf
            suma_g[intensidad] += gf
            suma_b[intensidad] += bf

    max_votos = 0
    nivel_ganador = 0
    
    for i in range(niveles):
        if conteos[i] > max_votos:
            max_votos = conteos[i]
            nivel_ganador = i

    if max_votos == 0:
        return 0.0, 0.0, 0.0
        
    r_final = suma_r[nivel_ganador] / max_votos
    g_final = suma_g[nivel_ganador] / max_votos
    b_final = suma_b[nivel_ganador] / max_votos
    
    return r_final, g_final, b_final

def aberracionCromatica(frame, y, x):

    alto, ancho, _ = frame.shape
    
    # Calculamos de dónde vamos a robar el color rojo y el azul
    x_rojo = max(0, x - desplazamiento)
    x_azul = min(ancho - 1, x + desplazamiento)
    
    # El Rojo viene de la izquierda, el Verde del centro, el Azul de la derecha
    r_final = frame[y, x_rojo, 0]
    g_final = frame[y, x, 1]
    b_final = frame[y, x_azul, 2]
    
    return r_final, g_final, b_final