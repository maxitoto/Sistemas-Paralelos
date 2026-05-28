# Sistemas Paralelos

**Autor:** Tomás Blanco  
**Cátedra:** Sistemas Paralelos – Federico Gonzales – UNTDF  

## 1. Abstract

This paper critically analyzes the impact of thread topological configuration (threads per block) on the performance of parallel image processing algorithms —specifically grayscale conversion and edge detection using the Sobel filter— executed on Graphics Processing Units (GPUs) using CUDA and Numba. It explores the close relationship between block dimensions and execution cycle efficiency, based on the architecture of warps (indivisible groups of 32 threads), demonstrating how misaligned configurations generate hardware underutilization. Furthermore, the fundamental role of memory coalescing in accessing two-dimensional matrices is evaluated, providing empirical evidence that distributions such as (16, 16) maximize bandwidth by ensuring contiguous memory reads. Finally, the study addresses the physical limits of Streaming Multiprocessors, detailing how an excess of threads increases register pressure and reduces overall occupancy. It concludes that optimal parallelism does not adhere to static configurations or the maximum allowed threads, but depends intrinsically on the underlying GPU microarchitecture and requires empirical profiling (benchmarking) for proper calibration.

**Keywords:** CUDA, Numba, GPU, Warps, Memory Coalescing, Occupancy.

## 2. Introducción

Este trabajo analiza críticamente el impacto de la configuración topológica de hilos (threads per block) en el rendimiento de algoritmos paralelos aplicados al procesamiento digital de imágenes —específicamente en la conversión a escala de grises y la detección de bordes mediante el filtro de Sobel— ejecutados sobre unidades de procesamiento gráfico (GPU) utilizando CUDA y Numba. Se explora la estrecha relación entre las dimensiones del bloque y la eficiencia de los ciclos de ejecución, fundamentada en la arquitectura de warps (grupos indivisibles de 32 hilos), demostrando cómo las configuraciones no alineadas generan subutilización de hardware. Asimismo, se evalúa el rol fundamental de la coalescencia de memoria en el acceso a matrices bidimensionales, evidenciando empíricamente que distribuciones como (16, 16) maximizan el ancho de banda al garantizar lecturas contiguas. Finalmente, el estudio aborda los límites físicos de los Multiprocesadores de Flujo (Streaming Multiprocessors), detallando cómo el exceso de hilos incrementa la presión de registros y reduce la ocupación (occupancy). Se concluye que el paralelismo óptimo no obedece a configuraciones estáticas o al máximo de hilos permitidos, sino que depende intrínsecamente de la microarquitectura de la GPU subyacente y requiere de perfilado empírico (benchmarking) para su correcta calibración.

**Palabras clave:** CUDA, Numba, GPU, Warps, Coalescencia de Memoria, Ocupación.

## 3. Metodología

Para evaluar el impacto de la configuración topológica de hilos (*threads per block*) frente a la ejecución secuencial, se diseñó un entorno de *benchmarking* estricto y automatizado. Las pruebas se ejecutaron utilizando una única imagen de entrada de **6000 x 6000 píxeles** (36 millones de píxeles por canal). Esta resolución masiva fue seleccionada intencionalmente como una prueba de estrés para saturar la capacidad de cómputo y el ancho de banda de la memoria del hardware.

Para garantizar la precisión de las métricas y aislar el tiempo neto de procesamiento algorítmico, el *pipeline* de ejecución se segmentó en cuatro fases aisladas:

* **Fase 0: Preprocesamiento y *Warm-up* (No contabilizada)**
Esta etapa prepara el entorno de ejecución. Incluye la lectura de la imagen desde el disco, la conversión de la matriz a un formato manejable por la biblioteca (`np.float32`) y, críticamente, la transferencia de los datos desde la memoria RAM (Host) hacia la memoria VRAM de la tarjeta gráfica (Device). Además, se realiza una ejecución "en vacío" (*warm-up*) de los kernels de Numba para forzar la compilación *Just-In-Time* (JIT). **El tiempo de esta fase se excluye del benchmark** para evitar que las latencias del bus PCIe y la compilación inicial contaminen las métricas de rendimiento del algoritmo.
* **Fase 1: Conversión a Escala de Grises (Contabilizada)**
Comienza la medición estricta de tiempo. La GPU ejecuta el kernel diseñado para promediar los canales RGB de la matriz, calculando la luminancia de cada píxel de forma masivamente paralela. Al finalizar, se sincronizan los hilos (`cuda.synchronize()`) y se detiene el cronómetro.
* **Fase 2: Filtro de Sobel (Contabilizada)**
Se reinicia el cronómetro para medir la convolución espacial. La GPU aplica las máscaras de Sobel ($G_x$ y $G_y$) sobre la matriz resultante de la Fase 1 para calcular la magnitud del gradiente y detectar los bordes. Nuevamente, se sincroniza el dispositivo y se registra el tiempo neto de cómputo.
* **Fase 3: Postprocesamiento y Descarga (No contabilizada)**
Una vez finalizados los cálculos matemáticos, la matriz resultante se transfiere de regreso desde la VRAM (Device) hacia la RAM del sistema (Host). Posteriormente, el procesador central (CPU) evalúa las métricas de validación (porcentaje de píxeles blancos) y reconstruye el artefacto visual guardando las imágenes resultantes en el disco duro. Al tratarse de operaciones de entrada/salida (I/O), **el tiempo de esta fase no se incluye en el cálculo del *Speed-up***.

**Configuraciones evaluadas:** El *pipeline* fue ejecutado bajo un modelo secuencial base (para establecer la métrica de referencia) y luego sometido a múltiples iteraciones paralelas sobre la GPU, variando dinámicamente las dimensiones del bloque de hilos (ej. 8x8, 16x16, 32x32) para identificar el punto óptimo de ocupación y rendimiento del hardware.

## 4. Implementación
[Repo en GitHub](https://github.com/maxitoto/Sistemas-Paralelos/tree/main/PipeLine)

## 5. Experimentos

Como un primer paso recomendado es realizar el descubrimiento de los limites físicos de la GPU.

```python
from numba import cuda

device = cuda.get_current_device()


print(device.MAX_THREADS_PER_BLOCK)
print(device.WARP_SIZE)
print(device.MAX_BLOCK_DIM_X)
print(device.MAX_BLOCK_DIM_Y)

#limites físicos 
MAX_THREADS_PER_BLOCK = 1024
WARP_SIZE = 32 
MAX_BLOCK_DIM_X = 1024 
MAX_BLOCK_DIM_Y = 1024
```

**WARP_SIZE:** Indica la unidad mínima e indivisible de ejecución de la GPU. La GPU agrupa y ejecuta por lotes de 32 al mismo tiempo y en estricta sincronía. Por lo tanto, indica matemáticamente que la cantidad total de hilos en tu bloque (X * Y) siempre debe ser un múltiplo de 32, si se usara 33 hilos, la GPU se ve obligada a usar dos warp (2x32 = 64) un primer warp totalmenete utilizado y un segundo warp con 31 hilos no usados.

**MAX_THREADS_PER_BLOCK:** Es el limite absoluto del volúmen de un bloque. Entonces, es la cantidad máxima de hilos que pueden convivir dentro de un mismo bloque, sin importar cómo los distribuyas geométricamente. 

Este es el límite de hardware puro. Si al multiplicar las dimensiones del bloque el resultado supera 1024 (por ejemplo, una configuración de 16x128 = 2048), mi GPU lanzara un error fatal en tiempo de ejecución **CUDA Out of Resources**

**MAX_BLOCK_DIM_X y MAX_BLOCK_DIM_Y:** Es el largo máximo que puede tener el eje X o el eje Y del bloque. No importa lo que quieras hacer el limite siempre será el **MAX_THREADS_PER_BLOCK** si quitas a uno le pones al otro. Ejemplo: 8x128 = 1024, pero 1024x1024 como ya mencione es un error conceptual.


**Información Previa:**

* **Contexto del Experimento:** Filtro Sobel 6000x6000
* **Hardware Detallado:** 
  - CPU: 
    - Intel64 Family 6 Model 158 Stepping 10 
    - 6 Cores 
    - OS Windows 
  - GPU: Nvidia GTX 1060 6GB 
    - Especificaciones CUDA: 
      - MAX_THREADS_PER_BLOCK: 1024
      - WARP_SIZE: 32
      - MAX_BLOCK_DIM_X: 1024
      - MAX_BLOCK_DIM_Y: 1024
- **Metodología:** Promedio de 5 iteraciones por método

**Resultados de la Ejecución:**

| Método | Threads per Block | T_RGB Promedio(s) | T_Sobel Promedio(s) | T_Total Promedio(s) | % Blancos | Speed-Up | Performance(%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **numba_gpu (8x16)** | 128 | 0.0106 | 0.0070 | **0.0176** | 0.0000 | **7568.85x** | 756884.6% |
| **numba_gpu (8x128)** | 1024 | 0.0111 | 0.0071 | **0.0183** | 0.0000 | **7272.09x** | 727209.1% |
| **numba_gpu (8x8)** | 64 | 0.0124 | 0.0079 | **0.0203** | 0.0000 | **6555.37x** | 655537.0% |
| **numba_gpu (16x16)** | 256 | 0.0152 | 0.0122 | **0.0274** | 0.0000 | **4852.25x** | 485224.9% |
| **numba_gpu (11x9)** | 99 | 0.0180 | 0.0101 | **0.0281** | 0.0000 | **4726.56x** | 472656.1% |
| **numba_gpu (16x8)** | 128 | 0.0173 | 0.0126 | **0.0299** | 0.0000 | **4439.20x** | 443919.6% |
| **numba_gpu (32x32)** | 1024 | 0.0177 | 0.0197 | **0.0374** | 0.0000 | **3550.30x** | 355029.7% |
| **numba_gpu (128x8)** | 1024 | 0.0190 | 0.0207 | **0.0397** | 0.0000 | **3349.78x** | 334977.6% |
| **secuencial (Baseline)** | - | 16.2594 | 116.5870 | **132.8464** | 0.0000 | **1.00x** | 100.0% |

---

## 6. Resultados

- Ganador: `(8, 16)` [128 hilos] - 0.0176s
  - **Observación:** Es la configuración más rápida de todo el experimento, batiendo incluso a bloques con más capacidad de procesamiento.
  - **El *Sweet Spot*:** Esta topología logra el equilibrio perfecto en la **Ocupación (*Occupancy*)**. Al solicitar solo 128 hilos por bloque, la presión de registros sobre los *Streaming Multiprocessors* (SM) de la GTX 1060 es extremadamente baja. Esto le permite al planificador de la GPU mantener múltiples bloques activos y "en vuelo" simultáneamente dentro del mismo SM. Cuando un bloque se frena esperando que llegue el color de un píxel desde la memoria RAM, el hardware cambia instantáneamente a otro bloque en nanosegundos, logrando que los núcleos CUDA jamás estén inactivos. Además, tener 16 hilos en el eje X (exactamente medio *warp*) garantiza transacciones de memoria muy limpias.

- Fuerza Bruta: `(8, 128)` [1024 hilos] - 0.0183s
  - **Observación:** A pesar de agotar el límite físico de la gráfica (1024 hilos), rinde casi tan rápido como el ganador.
  - **Saturación de Ancho de Banda:** Aquí triunfa la **Coalescencia de Memoria**. Al tener una anchura horizontal (eje X) masiva de 128 hilos, la GPU lee bloques gigantescos de píxeles contiguos en una sola pasada de barrido físico sobre la VRAM. Su única penalización (la razón por la que pierde frente a `8x16`) es que al instanciar 1024 hilos por bloque, asfixia los registros del procesador. El SM solo puede cargar este bloque y ninguno más. Queda segundo porque la brutal eficiencia al leer la memoria compensa casi por completo la pérdida de agilidad del procesador.

- Eficiente y Ligero: `(8, 8)` [64 hilos] - 0.0203s
  - **Observación:** Un bloque minúsculo que logra un tercer puesto excepcional.
  - **Alineación de Warps:** Un bloque de 64 hilos se traduce matemáticamente en **exactamente 2 *Warps*** (32 hilos por warp). No sobra ni falta un solo núcleo. La GTX 1060 ama esta simetría para su planificación interna. Se queda en tercer lugar porque al tener un ancho de lectura de solo 8 píxeles horizontales, la controladora de memoria debe hacer muchísimas más micropeticiones a la RAM para cubrir la imagen de 6000 píxeles de ancho, sin aprovechar al máximo el ancho de banda del bus.

- Estándar: `(16, 16)` [256 hilos] - 0.0274s
  - **Observación:** Aunque es el valor recomendado genéricamente en casi toda la literatura y libros (incluido tu profesor), en tu Filtro de Sobel cae al cuarto puesto.
  - **El efecto "Stencil" o Vecindad:** El Filtro de Sobel no procesa píxeles aislados; cada hilo necesita leer una vecindad de 3x3. Cuando usas bloques cuadrados perfectos y grandes como `16x16`, la relación matemática de "área vs. perímetro" cambia. Muchos hilos quedan en los "bordes" de este cuadrado 16x16 e intentan leer píxeles que físicamente le pertenecen a otro bloque vecino. Esto causa conflictos en la memoria caché L1 de la GTX 1060, haciendo que este bloque cuadrado rinda peor que uno más horizontal como el `8x16`.

- Desalineado de warp y miss de cache: `(11, 9)` [99 hilos] - 0.0281s
  - **Observación:** Rendimiento pobre para tener una cantidad de hilos (99) similar a los bloques rápidos.
  - ***Warp Divergence* y Fallo de Caché:** Esta topología es un "anti-patrón". Como 99 no es múltiplo de 32, el hardware se ve forzado a asignar 4 *warps* (128 espacios en total). El último *warp* tendrá a **29 de sus 32 núcleos literalmente apagados y ociosos**. Sumado a esto, un acceso de anchura `9` desalinea por completo las lecturas de memoria con las líneas de caché de 128 bytes de Nvidia, obligando a la tarjeta a leer datos basura que luego descartará.

- La fragmentación de lectura: `(16, 8)` [128 hilos] - 0.0299s
  - **Observación:** Tarda casi el doble que su gemelo transpuesto `(8, 16)`, a pesar de tener los mismos 128 hilos.
  - **Razón Teórica (Fragmentación de Lectura):** Dado que las matrices de imágenes se guardan en la memoria RAM en formato *Row-Major* (por filas contiguas), tener un ancho `X=8` significa que el grupo de hilos lee solo 8 píxeles antes de "chocar" y verse obligado a saltar físicamente de fila en la memoria. Esto rompe la **Coalescencia de Memoria** y duplica el esfuerzo electromecánico del hardware para traer la misma cantidad de datos.

- Asfixia: `(32, 32)` [1024 hilos] - 0.0374s
  - **Observación:** Es la forma cuadrada más grande soportada por el hardware, pero rinde terriblemente.
  - **Colapso por Registros:** Como vimos en `8x128`, lanzar 1024 hilos impide que el SM cambie de tarea. Pero a diferencia del `8x128` (que leía horizontalmente muy rápido), el `32x32` padece horriblemente del efecto "vecindad" del filtro de Sobel discutido en el punto 4. Al chocar los problemas de caché (por ser cuadrado) con la asfixia del SM (por ser masivo), la tarjeta pierde todo su poder de paralelismo.

- El Peor Escenario Posible: `(128, 8)` [1024 hilos] - 0.0397s
   - **Observación:** Es la configuración más lenta absoluta del experimento, tardando un 125% más de tiempo que la versión ganadora.
   - **El Peor Escenario:** Combina las dos peores ineficiencias de la arquitectura CUDA. Por un lado, bloquea la agilidad del procesador (Ocupación asfixiada por pedir 1024 hilos). Por otro lado, destruye el ancho de banda al pedirle a la controladora de memoria que baje por una "columna" extremadamente estrecha y profunda de la imagen (apenas 8 píxeles de ancho y 128 de alto). La GTX 1060 pasa la mayor parte de este tiempo simplemente esperando a que la VRAM busque y entregue datos fragmentados.


## 7. Conclusión
Este trabajo demuestra empíricamente la abismal superioridad de las arquitecturas masivamente paralelas (GPU) frente al procesamiento secuencial (CPU) para tareas intensivas como el filtrado espacial de imágenes de alta resolución. La delegación del cálculo computacional a la GPU logró reducir el tiempo de ejecución de una matriz de 36 millones de píxeles desde los casi 133 segundos hasta escasos 17 milisegundos, representando un *speed-up* superior a **7500x**. 

Sin embargo, el hallazgo más trascendental del experimento es que el máximo rendimiento en CUDA no se alcanza operando bajo la premisa intuitiva de "más hilos es mejor", ni tampoco adoptando ciegamente recetas teóricas generalistas. Un claro ejemplo de esto fue el comportamiento de la configuración de `16x16` (256 hilos): a pesar de ser el estándar y el punto de partida clásico en la industria, en este estudio quedó relegado exactamente a la mitad de la tabla de rendimiento de la GPU. Esto evidencia que los bloques estandarizados no contemplan las particularidades del algoritmo en cuestión; en el Filtro de Sobel, el acceso redundante a la vecindad de píxeles (*stencil*) genera una fricción en la memoria caché L1 que penaliza a las geometrías cuadradas perfectas en comparación con distribuciones más adaptadas. 

En su lugar, la configuración ganadora `(8, 16)` demostró que el verdadero punto óptimo (*sweet spot*) se halla en el equilibrio entre una baja presión de registros (alta ocupación de bloques simultáneos) y una disposición horizontal que favorezca la coalescencia. Por el contrario, forzar el límite máximo de 1024 hilos asfixia los multiprocesadores —como ocurrió en el `(32, 32)`— o destruye el ancho de banda si se ignora el orden *Row-Major* contiguo de las matrices en memoria —como en el fracaso del `(128, 8)`—. En definitiva, el desarrollo eficiente sobre GPU exige un diseño topológico estrictamente consciente del hardware subyacente, donde la alineación de *warps* y el patrón de acceso físico a la memoria se erigen como los verdaderos responsables del rendimiento computacional.

## 8. Recursos
[Explicación de Arquitectura de GPU](https://www.youtube.com/watch?v=whPSD8sdx-0)
