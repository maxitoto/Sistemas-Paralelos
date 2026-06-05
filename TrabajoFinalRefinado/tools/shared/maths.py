# tools/shared/maths.py

def oleo(kernel):
    """
    info util: https://docs.gimp.org/es/gimp-filter-oilify.html
    """
    niveles = 20  # Sensibilidad del pincel (cuántos tonos de pintura existen)
    
    # Listas para agrupar los colores según su nivel de luz
    conteos = [0] * niveles
    suma_r = [0.0] * niveles
    suma_g = [0.0] * niveles
    suma_b = [0.0] * niveles

    # evaluar a todos los vecinos
    for fila in kernel:
        for r, g, b in fila:
            
            # Calculamos el promedio de luz del píxel (0.0 a 1.0) 
            # y lo encasillamos en uno de los 20 "niveles"
            rf, gf, bf = float(r), float(g), float(b)
            intensidad = int( (((rf + gf + bf) / 3.0) / 255.0) * (niveles - 1) )
            
            # Votamos por ese nivel y sumamos sus colores
            conteos[intensidad] += 1
            suma_r[intensidad] += r
            suma_g[intensidad] += g
            suma_b[intensidad] += b

    # encontrar la moda
    max_votos = 0
    nivel_ganador = 0
    
    for i in range(niveles):
        if conteos[i] > max_votos:
            max_votos = conteos[i]
            nivel_ganador = i

    # calcular el color de la pincelada
    if max_votos == 0:
        return 0.0, 0.0, 0.0
        
    # El color final del píxel central es el promedio exacto 
    # de todos los vecinos que pertenecían al grupo ganador
    r_final = suma_r[nivel_ganador] / max_votos
    g_final = suma_g[nivel_ganador] / max_votos
    b_final = suma_b[nivel_ganador] / max_votos
    
    return r_final, g_final, b_final