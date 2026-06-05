# tools/shared/maths.py

def _filtro_grayscale(r, g, b):
    return (r * 0.299) + (g * 0.587) + (b * 0.114)

def _filtro_invert(r, g, b):
    return (255.0 - r), (255.0 - g), (255.0 - b)

def _filtro_sepia(r, g, b):
    tr = (r * 0.393) + (g * 0.769) + (b * 0.189)
    tg = (r * 0.349) + (g * 0.686) + (b * 0.168)
    tb = (r * 0.272) + (g * 0.534) + (b * 0.131)
    
    return (255.0 if tr > 255.0 else tr,
            255.0 if tg > 255.0 else tg,
            255.0 if tb > 255.0 else tb)

FILTROS_DISPONIBLES = {
    "grayscale": {"func": _filtro_grayscale, "canales_out": 1},
    "invert": {"func": _filtro_invert, "canales_out": 3},
    "sepia": {"func": _filtro_sepia, "canales_out": 3}
}

def obtener_filtro(nombre_filtro):
    """
    Devuelve un diccionario con la función pura y sus propiedades estructurales.
    """
    info_filtro = FILTROS_DISPONIBLES.get(nombre_filtro)
    
    if info_filtro is None:
        raise ValueError(f"❌ [Error] El filtro '{nombre_filtro}' no existe. Opciones: {list(FILTROS_DISPONIBLES.keys())}")
        
    return info_filtro