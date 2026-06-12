niveles = 4
radio = 2

batch_size_torch = 1 # haciendo muy especifica seleccion de tipo de dato para el gpu, logre hacer que unfold sea menos denso
# al utilizar float16 ne lugar de float32, pase de x4 a x2 pero aun así no consigo aumentar el tamaño del batch.


radio_torch = 2 # para el cpu, la funcion problematica es unfold
'''
¿Por qué PyTorch explotó pidiendo 28.7 GB por UN solo frame?
En tu archivo de PyTorch, el bucle ahora aísla 1 solo frame correctamente. Pero cuando ese frame llega a F.unfold, ocurre esto:Un radio de 8 significa una cuadrícula vecinal de $17 \times 17$ píxeles.Eso da un total de 289 vecinos a evaluar por cada píxel.PyTorch está diseñado para vectorizar todo al extremo. Para calcular todo a la vez, F.unfold hace copias físicas de la imagen.Si multiplicamos:3840 píxeles × 2160 píxeles × 3 canales (RGB) × 289 vecinos × 4 bytes (float32) = ¡Exactamente 28,764,979,200 bytes!
'''
desplazamiento = 25