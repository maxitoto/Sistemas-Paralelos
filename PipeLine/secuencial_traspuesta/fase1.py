from PIL import Image as pil
import os
# fase 1 encontrar la imagen.
def procesar(imagen_path, config):

    if (not os.path.exists(imagen_path)):
        raise FileExistsError(f"Error: La imagen no fue encontrada en la ruta indicada \n {imagen_path}")
    
    try:
        imagen_ram = pil.open(imagen_path)
        
        #imagen_grises = imagen_pil.convert('L') # pillow tiene una funcion para llevar la imagen a escala de grises (no lo voy a usar)
        
        #alto y ancho de la imagen
        ancho, alto = imagen_ram.size

        pixeles = list(imagen_ram.getdata()) #obtenemos los pixeles de la imagen

        matriz_rgb = [[pixeles[f * ancho + c] for c in range(ancho)] for f in range(alto)] # llevamos la imagen/pixeles a una matriz donde es 3(Red Green Blue) x Ancho x Alto


        matrizGris = [[0 for _ in range(ancho)] for _ in range(alto)] # matriz de ceros para hacer el gris (mismo tamaño que la imagen)

        #recorrer cada pixel y hacerlo gris
        for fil in range(alto):
            for col in range(ancho):
                r = matriz_rgb[fil][col][0]
                g = matriz_rgb[fil][col][1]
                b =matriz_rgb[fil][col][2]

                gris = (r * 0.299) + (g * 0.587) + (b * 0.114)

                matrizGris[fil][col] = int(gris)

    except Exception as e:
        raise ValueError(f"Falla Crítica en Fase 1: No se pudo procesar el archivo. Error: {e}")
    
    
    return (matrizGris, alto, ancho)