# Sistemas Paralelos

**Autor:** Tomás Blanco

**Cátedra:** Sistemas Paralelos – Federico Gonzales – UNTDF

## 1. Abstract

This paper critically analyzes the impact of thread topological configuration (threads per block) on the performance of parallel image processing algorithms —specifically grayscale conversion and edge detection using the Sobel filter— executed on Graphics Processing Units (GPUs) using CUDA and Numba. It explores the close relationship between block dimensions and execution cycle efficiency, based on the architecture of warps (indivisible groups of 32 threads), demonstrating how misaligned configurations generate hardware underutilization. Furthermore, the fundamental role of memory coalescing in accessing two-dimensional matrices is evaluated, providing empirical evidence that distributions such as (16, 16) maximize bandwidth by ensuring contiguous memory reads. Finally, the study addresses the physical limits of Streaming Multiprocessors, detailing how an excess of threads increases register pressure and reduces overall occupancy. It concludes that optimal parallelism does not adhere to static configurations or the maximum allowed threads, but depends intrinsically on the underlying GPU microarchitecture and requires empirical profiling (benchmarking) for proper calibration.

**Keywords:** CUDA, Numba, GPU, Warps, Memory Coalescing, Occupancy.

## 2. Introducción

Este trabajo analiza críticamente el impacto de la configuración de hilos (threads per block) en el rendimiento de algoritmos paralelos aplicados al procesamiento digital de imágenes —específicamente en la conversión a escala de grises y la detección de bordes mediante el filtro de Sobel— ejecutados sobre unidades de procesamiento gráfico (GPU) utilizando CUDA y Numba. Se explora la estrecha relación entre las dimensiones del bloque y la eficiencia de los ciclos de ejecución, fundamentada en la arquitectura de warps (grupos indivisibles de 32 hilos), demostrando cómo las configuraciones no alineadas generan subutilización de hardware. Asimismo, se evalúa el rol fundamental de la coalescencia de memoria en el acceso a matrices bidimensionales, evidenciando empíricamente que distribuciones como (16, 16) maximizan el ancho de banda al garantizar lecturas contiguas. Finalmente, el estudio aborda los límites físicos de los Multiprocesadores de Flujo (Streaming Multiprocessors), detallando cómo el exceso de hilos incrementa la presión de registros y reduce la ocupación (occupancy). Se concluye que el paralelismo óptimo no obedece a configuraciones estáticas o al máximo de hilos permitidos, sino que depende intrínsecamente de la microarquitectura de la GPU subyacente y requiere de perfilado empírico (benchmarking) para su correcta calibración.

**Palabras clave:** CUDA, Numba, GPU, Warps, Coalescencia de Memoria, Ocupación.

## 3. Metodología

Para evaluar el impacto de la configuración topológica de hilos (_threads per block_) frente a la ejecución secuencial, se diseñó un entorno de _benchmarking_ estricto y automatizado. Las pruebas se ejecutaron utilizando una única imagen de entrada de **6000 x 6000 píxeles** (36 millones de píxeles por canal). Esta resolución masiva fue seleccionada intencionalmente como una prueba de estrés para saturar la capacidad de cómputo y el ancho de banda de la memoria del hardware.

Para garantizar la precisión de las métricas y aislar el tiempo neto de procesamiento algorítmico, el _pipeline_ de ejecución se segmentó en cuatro fases aisladas:

- **Fase 0: Preprocesamiento y _Warm-up_ (No contabilizada)**
  Esta etapa prepara el entorno de ejecución. Incluye la lectura de la imagen desde el disco, la conversión de la matriz a un formato manejable por la biblioteca (`np.float32`) y, críticamente, la transferencia de los datos desde la memoria RAM (Host) hacia la memoria VRAM de la tarjeta gráfica (Device). Además, se realiza una ejecución "en vacío" (_warm-up_) de los kernels de Numba para forzar la compilación _Just-In-Time_ (JIT). **El tiempo de esta fase se excluye del benchmark** para evitar que las latencias del bus PCIe y la compilación inicial contaminen las métricas de rendimiento del algoritmo.
- **Fase 1: Conversión a Escala de Grises (Contabilizada)**
  Comienza la medición estricta de tiempo. La GPU ejecuta el kernel diseñado para promediar los canales RGB de la matriz, calculando la luminancia de cada píxel de forma masivamente paralela. Al finalizar, se sincronizan los hilos (`cuda.synchronize()`) y se detiene el cronómetro.
- **Fase 2: Filtro de Sobel (Contabilizada)**
  Se reinicia el cronómetro para medir la convolución espacial. La GPU aplica las máscaras de Sobel ($G_x$ y $G_y$) sobre la matriz resultante de la Fase 1 para calcular la magnitud del gradiente y detectar los bordes. Nuevamente, se sincroniza el dispositivo y se registra el tiempo neto de cómputo.
- **Fase 3: Postprocesamiento y Descarga (No contabilizada)**
  Una vez finalizados los cálculos matemáticos, la matriz resultante se transfiere de regreso desde la VRAM (Device) hacia la RAM del sistema (Host). Posteriormente, el procesador central (CPU) evalúa las métricas de validación (porcentaje de píxeles blancos) y reconstruye el artefacto visual guardando las imágenes resultantes en el disco duro. Al tratarse de operaciones de entrada/salida (I/O), **el tiempo de esta fase no se incluye en el cálculo del _Speed-up_**.

**Configuraciones evaluadas:** El _pipeline_ fue ejecutado bajo un modelo secuencial base (para establecer la métrica de referencia) y luego sometido a múltiples iteraciones paralelas sobre la GPU, variando dinámicamente las dimensiones del bloque de hilos (ej. 8x8, 16x16, 32x32) para identificar el punto óptimo de ocupación y rendimiento del hardware.

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

**WARP_SIZE:** Indica la unidad mínima e indivisible de ejecución de la GPU. La GPU agrupa y ejecuta por lotes de 32 al mismo tiempo y en estricta sincronía. Por lo tanto, indica matemáticamente que la cantidad total de hilos en tu bloque (X \* Y) siempre debe ser un múltiplo de 32, si se usara 33 hilos, la GPU se ve obligada a usar dos warp (2x32 = 64) un primer warp totalmenete utilizado y un segundo warp con 31 hilos no usados.

**MAX_THREADS_PER_BLOCK:** Es el limite absoluto del volúmen de un bloque. Entonces, es la cantidad máxima de hilos que pueden convivir dentro de un mismo bloque, sin importar cómo los distribuyas geométricamente.

Este es el límite de hardware puro. Si al multiplicar las dimensiones del bloque el resultado supera 1024 (por ejemplo, una configuración de 16x128 = 2048), mi GPU lanzara un error fatal en tiempo de ejecución **CUDA Out of Resources**

**MAX_BLOCK_DIM_X y MAX_BLOCK_DIM_Y:** Es el largo máximo que puede tener el eje X o el eje Y del bloque. No importa lo que quieras hacer el limite siempre será el **MAX_THREADS_PER_BLOCK** si quitas a uno le pones al otro. Ejemplo: 8x128 = 1024, pero 1024x1024 como ya mencione es un error conceptual.

**Información Previa:**

- **Contexto del Experimento:** Filtro Sobel 6000x6000
- **Hardware Detallado:**
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

| Método                 | Threads per Block | T_RGB Promedio(s) | T_Sobel Promedio(s) | T_Total Computo(s) | % Blancos | Speed-Up      | Performance(%) |
| ---------------------- | ----------------- | ----------------- | ------------------- | ------------------ | --------- | ------------- | -------------- |
| **numba_gpu (8x128)**  | 1024              | 0.0059            | 0.0071              | **0.0130**         | 0.0000    | **10216.23x** | 1021622.8%     |
| **numba_gpu (8x16)**   | 128               | 0.0078            | 0.0069              | **0.0148**         | 0.0000    | **9027.05x**  | 902704.6%      |
| **numba_gpu (8x8)**    | 64                | 0.0095            | 0.0079              | **0.0174**         | 0.0000    | **7637.76x**  | 763776.4%      |
| **numba_gpu (16x16)**  | 256               | 0.0099            | 0.0122              | **0.0221**         | 0.0000    | **6038.51x**  | 603851.3%      |
| **numba_gpu (11x9)**   | 99                | 0.0127            | 0.0101              | **0.0229**         | 0.0000    | **5826.55x**  | 582654.7%      |
| **numba_gpu (16x8)**   | 128               | 0.0124            | 0.0125              | **0.0249**         | 0.0000    | **5351.90x**  | 535190.3%      |
| **numba_gpu (32x32)**  | 1024              | 0.0128            | 0.0196              | **0.0324**         | 0.0000    | **4109.48x**  | 410947.6%      |
| **numba_gpu (128x8)**  | 1024              | 0.0161            | 0.0194              | **0.0355**         | 0.0000    | **3748.57x**  | 374856.7%      |
| **numba_cpu_paralelo** | -                 | 0.4024            | 0.2450              | **0.6473**         | 0.0000    | **205.78x**   | 20577.9%       |
| **secuencial (Base)**  | -                 | 16.1368           | 117.0681            | **133.2049**       | 0.0000    | **1.00x**     | 100.0%         |

---

## 6. Resultados

- Ganador: `(8, 128)` [1024 hilos] - 0.0130s
  - **Observación:** Esta topología el la mejor al reducir el tiempo de cómputo a apenas 13 milisegundos, obteniendo un _speed-up_ superior a 10.000x. (locura)
  - **El triunfo de la Coalescencia:** Aunque lanzar 1024 hilos asfixia la capacidad de registros del Multiprocesador (SM) y reduce su ocupación (_Occupancy_), en el procesamiento de imágenes el verdadero cuello de botella es el ancho de banda. Al configurar un ancho horizontal masivo (`X=128`), la GPU logra una coalescencia de memoria perfecta, leyendo enormes fragmentos contiguos de la matriz en una única transacción de hardware. La asombrosa eficiencia de lectura compensa con creces cualquier penalización de la CPU gráfica.

- Sweet Spot: `(8, 16)` [128 hilos] - 0.0148s
  - **Observación:** Una configuración altamente optimizada que logra el segundo lugar con un rendimiento excepcional.
  - **El _Sweet Spot_ de Ocupación:** A diferencia del ganador, esta topología solicita solo 128 hilos, lo que impone una presión de registros ínfima. Esto le permite al planificador de la GPU mantener múltiples bloques "en vuelo" simultáneamente dentro del mismo SM ocultando la latencia de memoria. Sin embargo, en esta escala de 6000x6000, la estrategia matemática de mantener a los núcleos ocupados quedó ligeramente por detrás de la fuerza bruta de leer la memoria en bloques horizontales gigantescos del `(8, 128)`.

- Eficiente y Ligero: `(8, 8)` [64 hilos] - 0.0174s
  - **Observación:** Un bloque minúsculo que logra un tercer puesto excepcional.
  - **Alineación de Warps:** Un bloque de 64 hilos se traduce matemáticamente en **exactamente 2 _Warps_** (32 hilos por warp). No sobra ni falta un solo núcleo. La gráfica ama esta simetría para su planificación interna. Su rendimiento es menor que los líderes porque su estrechez horizontal (`X=8`) fuerza a la controladora a hacer más micropeticiones a la RAM de video -

- Estándar: `(16, 16)` [256 hilos] - 0.0221s
  - **Observación:** Aunque es el valor recomendado generalmente, en el Filtro de Sobel cae al cuarto puesto.
  - **El efecto "Stencil" o Vecindad:** El Filtro de Sobel no procesa píxeles aislados; cada hilo necesita leer una vecindad de 3x3. Cuando usas bloques cuadrados perfectos y grandes como `16x16`, muchos hilos quedan en los "bordes" de este cuadrado e intentan leer píxeles que físicamente le pertenecen a otro bloque vecino. Esto causa fricción en la memoria caché L1 de la gráfica, haciéndola menos eficiente que formas rectangulares apaisadas.

- Desalineado de warp y miss de cache: `(11, 9)` [99 hilos] - 0.0229s
  - **Observación:** Rendimiento pobre para tener una cantidad de hilos (99) similar a los bloques rápidos.
  - **_Warp Divergence_ y Fallo de Caché:** Esta topología es un "anti-patrón". Como 99 no es múltiplo de 32, el hardware asigna 4 _warps_ (128 espacios). El último _warp_ tendrá a **29 de sus 32 núcleos literalmente apagados y ociosos**. Además, una anchura de `9` desalinea las lecturas con las líneas de caché de 128 bytes de Nvidia.

- La fragmentación de lectura: `(16, 8)` [128 hilos] - 0.0249s
  - **Observación:** Tarda considerablemente más que su gemelo transpuesto `(8, 16)`, a pesar de tener los mismos 128 hilos.
  - **Motivo Probable:** Al estar la matriz guardada en formato de filas contiguas, tener un ancho `X=8` significa que el hilo lee muy pocos píxeles antes de verse obligado a saltar físicamente de fila en la memoria. Esto rompe la **Coalescencia de Memoria** y aumenta el esfuerzo electromecánico de la tarjeta. **_a checkar! me ví obligado a pedirlo a la IA_**

- Asfixia: `(32, 32)` [1024 hilos] - 0.0324s
  - **Observación:** Es la forma cuadrada más grande soportada, pero rinde pobremente frente a las demás alternativas.
  - **Colapso por Registros y Vecindad:** Al instanciar 1024 hilos, el SM se satura y no puede cambiar de bloque. Sumado a esto, hereda de forma exponencial el problema de "vecindad del caché L1" visto en la topología `(16, 16)`. Pierde todo su agilidad.

- El Peor Escenario Posible de la GPU: `(128, 8)` [1024 hilos] - 0.0355s
  - **Observación:** Es la configuración más lenta absoluta del experimento en GPU.
  - **Combina Ineficiencias:** Destruye la Ocupación (al exigir el límite de 1024 hilos) y simultáneamente destruye la Coalescencia de Memoria al tener un eje X de apenas 8 píxeles. La GPU pasa la mayor parte del tiempo ahogada esperando fragmentos de memoria.

- La limitación arquitectónica de la CPU: `numba_cpu_paralelo` - 0.6473s
  - **Observación:** La versión concurrente aprovechando múltiples núcleos del procesador intel (205x más rápida que el secuencial).
  - **SIMD vs SISD:** Aunque la paralelización en CPU es admirable, sigue tardando casi un segundo completo. La peor y más ineficiente configuración de la tarjeta gráfica `(128, 8)` es 18 veces más rápida que todo el procesador Intel trabajando al 100%. Esto evidencia la arquitectura especializada de las GPU para operaciones matemáticas vectoriales.

## 7. Conclusión

Este trabajo demuestra empíricamente la abismal superioridad de las arquitecturas masivamente paralelas (GPU) frente al procesamiento secuencial o multihilo tradicional en CPU para tareas intensivas como el filtrado espacial de imágenes de alta resolución. La delegación del cálculo computacional a la GPU logró reducir el tiempo de ejecución de una matriz de 36 millones de píxeles desde los 133 segundos de la ejecución secuencial base hasta escasos 13 milisegundos, representando un _speed-up_ superior a **10.000x** (que creo que es un monton).

Sin embargo, el hallazgo más trascendental del experimento es que el máximo rendimiento en CUDA no se alcanza operando bajo la premisa intuitiva de "más hilos es mejor", ni tampoco adoptando ciegamente recetas teóricas generalistas. Un claro ejemplo de esto fue el comportamiento de la configuración de `16x16` (256 hilos): a pesar de ser el estándar y el punto de partida clásico en la industria, en este estudio quedó relegado exactamente a la mitad de la tabla de rendimiento. Esto evidencia que los bloques estandarizados no contemplan las particularidades del algoritmo en cuestión; en el Filtro de Sobel, el acceso redundante a la vecindad de píxeles (_stencil_) genera una fricción en la memoria caché L1 que penaliza a las geometrías cuadradas en comparación con distribuciones más adaptadas.

En su lugar, la configuración ganadora `(8, 128)` redefinió las prioridades de diseño. Al saturar los hilos en el eje X, demostró que en algoritmos fuertemente condicionados por el acceso a la matriz (como el procesamiento de imágenes), la **Coalescencia de Memoria** es el factor más crítico. La eficiencia de leer bloques horizontales gigantescos superó holgadamente a topologías como la de `(8, 16)`, la cual, aunque mantenía una mejor ocupación interna del procesador (menor presión de registros), no lograba explotar el bus de memoria con la misma ferocidad. Por el contrario, ignorar el orden _Row-Major_ contiguo de las matrices en memoria desemboca en fracasos rotundos, como se evidenció en el desplome de rendimiento de la topología transpuesta `(128, 8)`. En definitiva, el desarrollo eficiente sobre GPU exige un diseño topológico estrictamente consciente del hardware subyacente, donde el patrón de acceso físico a la memoria y la alineación de _warps_ dictan las reglas definitivas del rendimiento computacional.

## Apendice A

Dado que me fui por las ramas respecto al trabajo pedido original, voy a retomar con la información solicitada sobre los tiempos de transferencia, comparación con numba CPU paralelo y tendencias sobre el speedup de numba GPU.

### Todas las tablas

---

#### Contexto del Experimento: Filtro Sobel 6000x6000

| Método                 | Threads per Block | T_RGB Promedio(s) | T_Sobel Promedio(s) | T_Total Computo(s) | % Blancos | Speed-Up      | Performance(%) |
| ---------------------- | ----------------- | ----------------- | ------------------- | ------------------ | --------- | ------------- | -------------- |
| **numba_gpu (8x128)**  | 1024              | 0.0059            | 0.0071              | **0.0130**         | 0.0000    | **10216.23x** | 1021622.8%     |
| **numba_gpu (8x16)**   | 128               | 0.0078            | 0.0069              | **0.0148**         | 0.0000    | **9027.05x**  | 902704.6%      |
| **numba_gpu (8x8)**    | 64                | 0.0095            | 0.0079              | **0.0174**         | 0.0000    | **7637.76x**  | 763776.4%      |
| **numba_gpu (16x16)**  | 256               | 0.0099            | 0.0122              | **0.0221**         | 0.0000    | **6038.51x**  | 603851.3%      |
| **numba_gpu (11x9)**   | 99                | 0.0127            | 0.0101              | **0.0229**         | 0.0000    | **5826.55x**  | 582654.7%      |
| **numba_gpu (16x8)**   | 128               | 0.0124            | 0.0125              | **0.0249**         | 0.0000    | **5351.90x**  | 535190.3%      |
| **numba_gpu (32x32)**  | 1024              | 0.0128            | 0.0196              | **0.0324**         | 0.0000    | **4109.48x**  | 410947.6%      |
| **numba_gpu (128x8)**  | 1024              | 0.0161            | 0.0194              | **0.0355**         | 0.0000    | **3748.57x**  | 374856.7%      |
| **numba_cpu_paralelo** | -                 | 0.4024            | 0.2450              | **0.6473**         | 0.0000    | **205.78x**   | 20577.9%       |
| **secuencial (Base)**  | -                 | 16.1368           | 117.0681            | **133.2049**       | 0.0000    | **1.00x**     | 100.0%         |

---

#### Contexto del Experimento: Filtro Sobel 3000x3000

| Método                 | Threads per Block | T_RGB Promedio(s) | T_Sobel Promedio(s) | T_Total Computo(s) | % Blancos | Speed-Up     | Performance(%) |
| ---------------------- | ----------------- | ----------------- | ------------------- | ------------------ | --------- | ------------ | -------------- |
| **numba_gpu (8x16)**   | 128               | 0.0020            | 0.0018              | **0.0038**         | 0.0000    | **8883.97x** | 888397.0%      |
| **numba_gpu (8x8)**    | 64                | 0.0020            | 0.0018              | **0.0039**         | 0.0000    | **8751.83x** | 875183.0%      |
| **numba_gpu (8x128)**  | 1024              | 0.0018            | 0.0021              | **0.0039**         | 0.0000    | **8689.25x** | 868925.3%      |
| **numba_gpu (11x9)**   | 99                | 0.0027            | 0.0028              | **0.0054**         | 0.0000    | **6209.20x** | 620920.2%      |
| **numba_gpu (16x16)**  | 256               | 0.0028            | 0.0036              | **0.0064**         | 0.0000    | **5323.02x** | 532302.4%      |
| **numba_gpu (16x8)**   | 128               | 0.0029            | 0.0036              | **0.0064**         | 0.0000    | **5269.06x** | 526905.7%      |
| **numba_gpu (128x8)**  | 1024              | 0.0039            | 0.0056              | **0.0095**         | 0.0000    | **3556.64x** | 355664.0%      |
| **numba_gpu (32x32)**  | 1024              | 0.0037            | 0.0059              | **0.0096**         | 0.0000    | **3523.14x** | 352314.4%      |
| **numba_cpu_paralelo** | -                 | 0.2537            | 0.2084              | **0.4621**         | 0.0014    | **73.21x**   | 7321.4%        |
| **secuencial (Base)**  | -                 | 4.0425            | 29.7903             | **33.8328**        | 0.0014    | **1.00x**    | 100.0%         |

---

#### Contexto del Experimento: Filtro Sobel 1500x1500

| Método                 | Threads per Block | T_RGB Promedio(s) | T_Sobel Promedio(s) | T_Total Computo(s) | % Blancos | Speed-Up     | Performance(%) |
| ---------------------- | ----------------- | ----------------- | ------------------- | ------------------ | --------- | ------------ | -------------- |
| **numba_gpu (8x8)**    | 64                | 0.0006            | 0.0005              | **0.0011**         | 0.0000    | **7617.31x** | 761731.5%      |
| **numba_gpu (8x16)**   | 128               | 0.0006            | 0.0005              | **0.0011**         | 0.0000    | **7476.84x** | 747684.3%      |
| **numba_gpu (8x128)**  | 1024              | 0.0006            | 0.0006              | **0.0012**         | 0.0000    | **7213.19x** | 721319.1%      |
| **numba_gpu (11x9)**   | 99                | 0.0008            | 0.0007              | **0.0015**         | 0.0000    | **5603.46x** | 560346.1%      |
| **numba_gpu (16x8)**   | 128               | 0.0009            | 0.0009              | **0.0018**         | 0.0000    | **4829.97x** | 482997.2%      |
| **numba_gpu (16x16)**  | 256               | 0.0009            | 0.0009              | **0.0018**         | 0.0000    | **4788.44x** | 478844.1%      |
| **numba_gpu (32x32)**  | 1024              | 0.0011            | 0.0015              | **0.0025**         | 0.0000    | **3386.05x** | 338605.4%      |
| **numba_gpu (128x8)**  | 1024              | 0.0011            | 0.0014              | **0.0025**         | 0.0000    | **3383.15x** | 338314.6%      |
| **numba_cpu_paralelo** | -                 | 0.2092            | 0.2032              | **0.4125**         | 0.0600    | **20.82x**   | 2081.9%        |
| **secuencial (Base)**  | -                 | 1.0229            | 7.5644              | **8.5873**         | 0.0600    | **1.00x**    | 100.0%         |

---

#### Contexto del Experimento: Filtro Sobel 750x750

| Método                 | Threads per Block | T_RGB Promedio(s) | T_Sobel Promedio(s) | T_Total Computo(s) | % Blancos | Speed-Up     | Performance(%) |
| ---------------------- | ----------------- | ----------------- | ------------------- | ------------------ | --------- | ------------ | -------------- |
| **numba_gpu (8x8)**    | 64                | 0.0002            | 0.0002              | **0.0004**         | 0.0001    | **5044.93x** | 504492.7%      |
| **numba_gpu (8x16)**   | 128               | 0.0002            | 0.0002              | **0.0004**         | 0.0001    | **5000.25x** | 500024.9%      |
| **numba_gpu (8x128)**  | 1024              | 0.0003            | 0.0002              | **0.0005**         | 0.0001    | **4166.15x** | 416614.7%      |
| **numba_gpu (11x9)**   | 99                | 0.0003            | 0.0002              | **0.0006**         | 0.0001    | **3805.77x** | 380577.0%      |
| **numba_gpu (16x16)**  | 256               | 0.0003            | 0.0003              | **0.0006**         | 0.0001    | **3771.87x** | 377187.2%      |
| **numba_gpu (16x8)**   | 128               | 0.0003            | 0.0003              | **0.0007**         | 0.0001    | **3214.48x** | 321447.9%      |
| **numba_gpu (128x8)**  | 1024              | 0.0003            | 0.0004              | **0.0007**         | 0.0001    | **2816.69x** | 281668.9%      |
| **numba_gpu (32x32)**  | 1024              | 0.0004            | 0.0004              | **0.0008**         | 0.0001    | **2647.61x** | 264761.3%      |
| **numba_cpu_paralelo** | -                 | 0.1976            | 0.2058              | **0.4034**         | 0.2818    | **5.21x**    | 520.7%         |
| **secuencial (Base)**  | -                 | 0.2490            | 1.8514              | **2.1004**         | 0.2818    | **1.00x**    | 100.0%         |

### Analisis

- ¿Qué mejora de tiempo se observa entre Numba GPU y Numba paralelo CPU para cada tamaño de imagen?
  Para evaluar la mejora real de procesamiento, lo voy a comparar con los mejores combinaciones (x, y) en cada imagen, esto es para tirar todo el trabajo ya hecho.

  | Resolución    | Numba CPU Paralelo | Numba GPU (Mejor Cómputo) | Mejora (Aceleración relativa) |
  | ------------- | ------------------ | ------------------------- | ----------------------------- |
  | **750x750**   | 0.4034 s           | 0.0004 s _(8x8)_          | **~1008x más rápido**         |
  | **1500x1500** | 0.4125 s           | 0.0011 s _(8x8)_          | **~375x más rápido**          |
  | **3000x3000** | 0.4621 s           | 0.0038 s _(8x16)_         | **~121x más rápido**          |
  | **6000x6000** | 0.6473 s           | 0.0130 s _(8x128)_        | **~49x más rápido**           |

  Se observa una superioridad de la GPU en todos los escenarios. Sin embargo, lo interesante ocurre en las resoluciones más bajas (750x750) ya que la velocidad realativa entre CPU y GPU es más de 1000 veces más rápido. Esto ocurre porque **el CPU Paralelo presenta un _overhead_ estático muy alto** (tarda alrededor de 0.40 segundos sin importar el tamaño de la imagen, presumiblemente por el costo que tiene el sistema operativo para instanciar y sincronizar los hilos del procesador). La GPU, por el contrario, ejecuta el cálculo en microsegundos, haciendo que la diferencia relativa sea astronómica en cargas pequeñas.

- ¿En qué tamaños de imagen se amortiza el costo de transferencia de datos CPU<->GPU?

  Para que el uso de la GPU esté justificado, el tiempo total de la GPU (Transferencia IN + Cómputo + Transferencia OUT) debe ser menor que el tiempo que le tomaría a la CPU resolver el problema sin mover los datos.

  Tomando el escenario más pequeño (**750x750**), porque sería el más rápido para el CPU:
  - **Tiempo Total GPU:** ~0.0015s (IN) + 0.0004s (Cómputo) + 0.0005s (OUT) = **0.0024 segundos**.
  - **Tiempo Total CPU Paralelo:** **0.4034 segundos**.
  - **Tiempo Secuencial:** **2.1004 segundos**.

  **Conclusión:** El costo de transferencia de datos **se amortiza en absolutamente todos los tamaños de imagen evaluados**, incluso en el más pequeño (750x750), donde el flujo completo a través del bus PCIe sigue siendo más de 160 veces más rápido que la CPU en paralelo.
  No obstante, los datos revelan que **la transferencia es el verdadero cuello de botella de la arquitectura GPU**. Por ejemplo, en 6000x6000, mover los datos de ida y vuelta toma ~0.078s, mientras que calcularlos toma solo 0.013s. Es decir, la GPU pasa el 85% de su tiempo esperando que los datos viajen por la placa base y solo un 15% haciendo matemáticas.

- ¿El comportamiento del speed-up de Numba GPU es consistente al aumentar la resolución, o presenta umbrales/cambios de tendencia?

  El comportamiento del _Speed-up_ neto frente al algoritmo secuencial **no es consistente (plano), sino que presenta una clara tendencia ascendente** a medida que aumenta la carga de trabajo.

  Observando las mejores configuraciones por resolución:
  - **750x750:** Speed-up máximo de **~5044x**
  - **1500x1500:** Speed-up máximo de **~7617x**
  - **3000x3000:** Speed-up máximo de **~8883x**
  - **6000x6000:** Speed-up máximo de **~10216x**

  **El cambio de tendencia:** Esta curva ascendente demuestra el concepto de **Saturación y Ocupación del Hardware**. En las imágenes pequeñas (750x750, apenas medio millón de píxeles), la tarjeta gráfica sufre de _"Starvation"_ (inanición): tiene miles de núcleos disponibles, pero no hay suficiente trabajo para mantenerlos a todos ocupados simultáneamente. El _overhead_ de lanzar el _kernel_ diluye la eficiencia matemática.
  A medida que la imagen crece hasta los 36 millones de píxeles (6000x6000), la GPU finalmente recibe una carga lo suficientemente masiva para alimentar por completo todos sus _Streaming Multiprocessors_ al 100% de su capacidad. Esto demuestra que la arquitectura GPU requiere cargas masivas para alcanzar su verdadera eficiencia. El rendimiento relativo mejora exponencialmente a medida que se llenan los Streaming Multiprocessors, hasta llegar a un punto de saturación, donde el 100% del hardware está ocupado y la aceleración encuentra su techo máximo.

## Apendice B

Dado que había conceptos que no conocía, mi trabajo estaba comparando peras con manzanas. Digamos que en la implementación hace casteos de float32 a uint8 o viceversa donde no debía (a veces en la CPU, a veces en la GPU), y solo ese minúsculo cambio ya implica alterar los resultados. En el caso de hacerlo en la CPU, esto provoca que la GPU deba enviar datos en float32 por el medio. Esto lo logrará, pero es demasiado lento porque ocupa todo el bus. La solución es hacer el cast a uint8 dentro de la GPU y luego transferir a la CPU ¿De dónde viene el problema en mi implementación? Viene de desconocer PyTorch cuando aún trabajaba con Numba. Pytorch exige float32 y Numba no; al no saber cómo funciona una herramienta, el benchmarck/pipeline no está preparado. Así como este problema, tuve varios más (diminutos, pero importantes).

### Todas las tablas

### Sobel 750x750

|Método            |Threads Per Block|T_Transf_IN(s)|T_RGB_Promedio(s)|T_Sobel_Promedio(s)|T_Total_Computo(s)|T_Transf_OUT(s)|%_Blancos |Tiempo_Total(s)|Speed-Up_Computo|Speed-Up_Total|Performance(%)|
|------------------|-----------------|--------------|-----------------|-------------------|------------------|---------------|----------|---------------|----------------|--------------|--------------|
|numba_gpu_8_8     |64               |0.00150002    |0.00031106       |0.00045972         |0.00077078        |0.00030520     |0.28373333|0.00257600     |4034.19x        |1207.09x      |403419.4%     |
|numba_gpu_8_16    |128              |0.00150942    |0.00029092       |0.00046096         |0.00075188        |0.00034464     |0.28373333|0.00260594     |4135.60x        |1193.23x      |413560.1%     |
|numba_gpu_16_16   |256              |0.00143190    |0.00037702       |0.00048576         |0.00086278        |0.00039964     |0.28373333|0.00269432     |3604.02x        |1154.09x      |360401.9%     |
|numba_gpu_16_8    |128              |0.00155148    |0.00042814       |0.00048284         |0.00091098        |0.00032438     |0.28373333|0.00278684     |3413.33x        |1115.77x      |341333.0%     |
|numba_gpu_8_128   |1024             |0.00163252    |0.00031708       |0.00051438         |0.00083146        |0.00036816     |0.28373333|0.00283214     |3739.78x        |1097.92x      |373977.8%     |
|numba_gpu_32_32   |1024             |0.00159200    |0.00047404       |0.00053734         |0.00101138        |0.00032494     |0.28373333|0.00292832     |3074.49x        |1061.86x      |307448.8%     |
|numba_gpu_128_8   |1024             |0.00162924    |0.00035582       |0.00046114         |0.00081696        |0.00050970     |0.28373333|0.00295590     |3806.15x        |1051.96x      |380615.4%     |
|numba_gpu_11_9    |99               |0.00162288    |0.00043292       |0.00070634         |0.00113926        |0.00046230     |0.28373333|0.00322444     |2729.38x        |964.35x       |272938.2%     |
|numba_cpu_paralelo|-                |0.00000000    |0.00079402       |0.00278472         |0.00357874        |0.00000000     |0.28373333|0.00357874     |868.87x         |868.87x       |86887.4%      |
|pytorch_gpu       |-                |0.00135808    |0.00037432       |0.00551562         |0.00588994        |0.00057432     |0.28373333|0.00782234     |527.93x         |397.51x       |52793.0%      |
|pytorch_cpu       |-                |0.00000000    |0.00277694       |0.01437890         |0.01715584        |0.00000000     |0.28373333|0.01715584     |181.25x         |181.25x       |18124.9%      |
|numpy_cpu         |-                |0.00000000    |0.00600780       |0.02626344         |0.03227124        |0.00000000     |0.28373333|0.03227124     |96.35x          |96.35x        |9635.4%       |
|secuencial        |-                |0.00000000    |0.49513482       |2.61434094         |3.10947576        |0.00000000     |0.28373333|3.10947576     |1.00x           |1.00x         |100.0%        |

---

### Sobel 1500x1500

|Método            |Threads Per Block|T_Transf_IN(s)|T_RGB_Promedio(s)|T_Sobel_Promedio(s)|T_Total_Computo(s)|T_Transf_OUT(s)|%_Blancos |Tiempo_Total(s)|Speed-Up_Computo|Speed-Up_Total|Performance(%)|
|------------------|-----------------|--------------|-----------------|-------------------|------------------|---------------|----------|---------------|----------------|--------------|--------------|
|numba_gpu_128_8   |1024             |0.00473660    |0.00080978       |0.00164244         |0.00245222        |0.00145372     |0.06062222|0.00864254     |5116.46x        |1451.74x      |511645.9%     |
|numba_gpu_8_16    |128              |0.00498452    |0.00076198       |0.00167326         |0.00243524        |0.00138924     |0.06062222|0.00880900     |5152.13x        |1424.30x      |515213.4%     |
|numba_gpu_8_8     |64               |0.00474412    |0.00078132       |0.00167534         |0.00245666        |0.00169560     |0.06062222|0.00889638     |5107.21x        |1410.31x      |510721.2%     |
|numba_gpu_11_9    |99               |0.00485852    |0.00104918       |0.00207178         |0.00312096        |0.00133072     |0.06062222|0.00931020     |4020.14x        |1347.63x      |402013.6%     |
|numba_gpu_16_16   |256              |0.00510328    |0.00116400       |0.00164312         |0.00280712        |0.00145056     |0.06062222|0.00936096     |4469.59x        |1340.32x      |446959.3%     |
|numba_gpu_32_32   |1024             |0.00487106    |0.00135232       |0.00181332         |0.00316564        |0.00136444     |0.06062222|0.00940114     |3963.40x        |1334.59x      |396339.5%     |
|numba_gpu_16_8    |128              |0.00525474    |0.00119684       |0.00164216         |0.00283900        |0.00141706     |0.06062222|0.00951080     |4419.40x        |1319.20x      |441940.2%     |
|numba_gpu_8_128   |1024             |0.00499550    |0.00075988       |0.00171586         |0.00247574        |0.00263722     |0.06062222|0.01010846     |5067.85x        |1241.21x      |506785.2%     |
|numba_cpu_paralelo|-                |0.00000000    |0.00327600       |0.01144326         |0.01471926        |0.00000000     |0.06062222|0.01471926     |852.40x         |852.40x       |85239.9%      |
|pytorch_gpu       |-                |0.00455358    |0.00119356       |0.00719638         |0.00838994        |0.00221416     |0.06062222|0.01515768     |1495.44x        |827.74x       |149544.4%     |
|pytorch_cpu       |-                |0.00000000    |0.01165812       |0.06164418         |0.07330230        |0.00000000     |0.06062222|0.07330230     |171.16x         |171.16x       |17116.4%      |
|numpy_cpu         |-                |0.00000000    |0.02510874       |0.10843344         |0.13354218        |0.00000000     |0.06062222|0.13354218     |93.95x          |93.95x        |9395.3%       |
|secuencial        |-                |0.00000000    |2.02521430       |10.52146854        |12.54668284       |0.00000000     |0.06062222|12.54668284    |1.00x           |1.00x         |100.0%        |


---

### Sobel 3000x3000

|Método            |Threads Per Block|T_Transf_IN(s)|T_RGB_Promedio(s)|T_Sobel_Promedio(s)|T_Total_Computo(s)|T_Transf_OUT(s)|%_Blancos |Tiempo_Total(s)|Speed-Up_Computo|Speed-Up_Total|Performance(%)|
|------------------|-----------------|--------------|-----------------|-------------------|------------------|---------------|----------|---------------|----------------|--------------|--------------|
|numba_gpu_8_128   |1024             |0.01824062    |0.00241214       |0.00642246         |0.00883460        |0.00455084     |0.00141111|0.03162606     |5654.41x        |1579.54x      |565441.5%     |
|numba_gpu_8_8     |64               |0.01756520    |0.00357622       |0.00633122         |0.00990744        |0.00421398     |0.00141111|0.03168662     |5042.12x        |1576.52x      |504211.9%     |
|numba_gpu_8_16    |128              |0.01852676    |0.00312142       |0.00630574         |0.00942716        |0.00475238     |0.00141111|0.03270630     |5299.00x        |1527.37x      |529899.7%     |
|numba_gpu_16_16   |256              |0.01813030    |0.00396968       |0.00634228         |0.01031196        |0.00459478     |0.00141111|0.03303704     |4844.33x        |1512.08x      |484432.6%     |
|numba_gpu_16_8    |128              |0.01824558    |0.00430522       |0.00633206         |0.01063728        |0.00450130     |0.00141111|0.03338416     |4696.17x        |1496.35x      |469617.2%     |
|numba_gpu_128_8   |1024             |0.01858798    |0.00355604       |0.00636022         |0.00991626        |0.00531318     |0.00141111|0.03381742     |5037.63x        |1477.18x      |503763.4%     |
|numba_gpu_32_32   |1024             |0.01768804    |0.00476916       |0.00704170         |0.01181086        |0.00475682     |0.00141111|0.03425572     |4229.54x        |1458.28x      |422953.9%     |
|numba_gpu_11_9    |99               |0.01799222    |0.00466766       |0.00809780         |0.01276546        |0.00461590     |0.00141111|0.03537358     |3913.25x        |1412.20x      |391325.4%     |
|pytorch_gpu       |-                |0.01688272    |0.00435894       |0.01698602         |0.02134496        |0.00729792     |0.00141111|0.04552560     |2340.34x        |1097.28x      |234034.1%     |
|numba_cpu_paralelo|-                |0.00000000    |0.01310678       |0.04784488         |0.06095166        |0.00000000     |0.00141111|0.06095166     |819.58x         |819.58x       |81957.6%      |
|pytorch_cpu       |-                |0.00000000    |0.03621198       |0.21738500         |0.25359698        |0.00000000     |0.00141111|0.25359698     |196.98x         |196.98x       |19698.4%      |
|numpy_cpu         |-                |0.00000000    |0.09535462       |0.44133576         |0.53669038        |0.00000000     |0.00141111|0.53669038     |93.08x          |93.08x        |9307.9%       |
|secuencial        |-                |0.00000000    |8.00460592       |41.94988652        |49.95449244       |0.00000000     |0.00141111|49.95449244    |1.00x           |1.00x         |100.0%        |


---

### Sobel 6000x6000

|Método            |Threads Per Block|T_Transf_IN(s)|T_RGB_Promedio(s)|T_Sobel_Promedio(s)|T_Total_Computo(s)|T_Transf_OUT(s)|%_Blancos |Tiempo_Total(s)|Speed-Up_Computo|Speed-Up_Total|Performance(%)|
|------------------|-----------------|--------------|-----------------|-------------------|------------------|---------------|----------|---------------|----------------|--------------|--------------|
|numba_gpu_16_16   |256              |0.06793178    |0.01303250       |0.02048970         |0.03352220        |0.01522626     |0.00000000|0.11668024     |5916.05x        |1699.68x      |591604.8%     |
|numba_gpu_8_128   |1024             |0.07236560    |0.00769972       |0.02089888         |0.02859860        |0.01598386     |0.00000000|0.11694806     |6934.57x        |1695.79x      |693456.8%     |
|numba_gpu_32_32   |1024             |0.06537966    |0.01568842       |0.02261286         |0.03830128        |0.01557176     |0.00000000|0.11925270     |5177.87x        |1663.01x      |517786.7%     |
|numba_gpu_128_8   |1024             |0.06883432    |0.01810304       |0.02044600         |0.03854904        |0.01590582     |0.00000000|0.12328918     |5144.59x        |1608.57x      |514458.8%     |
|numba_gpu_8_8     |64               |0.06982756    |0.01852076       |0.02058976         |0.03911052        |0.01549826     |0.00000000|0.12443634     |5070.73x        |1593.74x      |507073.1%     |
|numba_gpu_16_8    |128              |0.06828876    |0.02011672       |0.02043624         |0.04055296        |0.01604554     |0.00000000|0.12488726     |4890.37x        |1587.98x      |489036.9%     |
|numba_gpu_8_16    |128              |0.07897506    |0.01182820       |0.02047472         |0.03230292        |0.01598860     |0.00000000|0.12726658     |6139.35x        |1558.30x      |613935.0%     |
|numba_gpu_11_9    |99               |0.07014582    |0.02152248       |0.03007966         |0.05160214        |0.01632160     |0.00000000|0.13806956     |3843.23x        |1436.37x      |384323.1%     |
|pytorch_gpu       |-                |0.05770878    |0.01687706       |0.05030086         |0.06717792        |0.06531458     |0.00000000|0.19020128     |2952.14x        |1042.68x      |295214.5%     |
|numba_cpu_paralelo|-                |0.00000000    |0.05270446       |0.18460628         |0.23731074        |0.00000000     |0.00000000|0.23731074     |835.69x         |835.69x       |83569.3%      |
|pytorch_cpu       |-                |0.00000000    |0.16665436       |0.91130114         |1.07795550        |0.00000000     |0.00000000|1.07795550     |183.98x         |183.98x       |18397.7%      |
|numpy_cpu         |-                |0.00000000    |0.37212598       |1.73352258         |2.10564856        |0.00000000     |0.00000000|2.10564856     |94.18x          |94.18x        |9418.4%       |
|secuencial        |-                |0.00000000    |32.37388468      |165.94505884       |198.31894352      |0.00000000     |0.00000000|198.31894352   |1.00x           |1.00x         |100.0%        |


### Analisis

- ¿Qué diferencias de rendimiento se observan entre PyTorch CPU y PyTorch GPU para cada tamaño de imagen?
  - Sobel 750x750:
    - PyTorch CPU: ~0.017 segundos (Speed-Up: 181.25x)
    - PyTorch GPU: ~0.005 segundos (Speed-Up: 527.93x)
    - Diferencia: La GPU es ~2.9 veces más rápida que la CPU.
  - Sobel 1500x1500:
    - PyTorch CPU: ~0.073 segundos (Speed-Up: 171.16x)
    - PyTorch GPU: ~0.008 segundos (Speed-Up: 1495.44x)
    - Diferencia: La GPU es ~8.7 veces más rápida que la CPU.
  - Sobel 3000x3000:
    - PyTorch CPU: ~0.253 segundos (Speed-Up: 196.98x)
    - PyTorch GPU: ~0.021 segundos (Speed-Up: 2340.34x)
    - Diferencia: La GPU es ~11.8 veces más rápida que la CPU.
  - Sobel 6000x6000:
    - PyTorch CPU: ~1.077 segundos (Speed-Up: 183.98x)
    - PyTorch GPU: ~0.067 segundos (Speed-Up: 2952.14x)
    - Diferencia: La GPU es ~16 veces más rápida que la CPU.

  La columna Speed-Up de PyTorch CPU en todos los experimentos es una línea casi plana. Oscila siempre entre los 171x y los 196x respecto al secuencial. Esto significa que el procesador en mi caso de 6 núcleos llega a su máxima capacidad muy rápido. Para la CPU, procesar una imagen gigante no la hace más eficiente, simplemente se le exige más y por más tiempo.
  Por otro lado, en la imagen pequeña de 750x750, la GPU apenas es 2.9 veces más rápida. Esto ocurre porque enviar una imagen tan chica a una tarjeta gráfica con 1024 hilos es un desperdicio; la mayoría de los núcleos se quedan osciosos, y el "overhead" (de preparar la tarjeta) ocupa gran parte del tiempo.
  Sin embargo, a medida que la imagen crece a 6000x6000 (36 millones de píxeles), la GPU por fin tiene suficiente trabajo masivo para alimentar y encender todos sus núcleos al mismo tiempo. Es por eso que su Speed-Up explota de 527x hasta casi 3000x, logrando despachar la imagen más pesada en tan solo 67 milisegundos.

- ¿Cómo se comparan PyTorch CPU/PyTorch GPU frente a las mejores implementaciones obtenidas en las entregas anteriores?
  - Respecto a CPUs vs CPUs:
    - PyTorch CPU: 1.0779 segundos (Speed-Up Cómputo: ~184x)
    - Numba CPU Paralelo: 0.2373 segundos (Speed-Up Cómputo: ~835x)
    Numba en CPU es ~4.5 veces más rápido que el motor de CPU de PyTorch.
    PyTorch en CPU no es tan eficiente porque utiliza librerías genéricas de C++ en el fondo. Además, al aplicar las dos convoluciones por separado (gx y gy), recorre la memoria de la RAM dos veces. Por el contrario, la directiva @njit(parallel=True) de Numba invoca al compilador LLVM de la máquina, el cual fusiona los cálculos en un solo recorrido de memoria altamente optimizado para la arquitectura exacta del procesador Intel.
  - Respecto a GPUs vs GPUs 
    - PyTorch GPU: 0.0671 segundos (Speed-Up Cómputo: ~2952x)
    - Mejor Numba GPU (8_128): 0.0285 segundos (Speed-Up Cómputo: ~6934x) 
    El kernel a medida en Numba GPU es ~2.35 veces más rápido que el motor cuDNN de PyTorch.
    De nuevo el triunfo de la Fusión de Kernels. Numba GPU hace la lectura, la matemática de Gx, Gy, la raíz cuadrada y la escritura en un solo viaje de memoria. PyTorch GPU necesita múltiples llamadas a la memoria global de la gráfica para guardar tensores intermedios, estrangulando el ancho de banda.
  - Pytorch es fácil de usar, pero bastante malo comparado con Numba personalizado.
    - Numba CPU Paralelo (Tiempo Total): 0.2373 segundos
    - PyTorch GPU (Tiempo Total): 0.1902 segundos (IN: 0.057s + Computo: 0.067s + OUT: 0.065s)
    El procesador tradicional (CPU) se apróxima a la tarjeta gráfica (GPU) dejando a PyTorch atrás! La diferencia en el mundo real es de apenas ~47 milisegundos a favor de PyTorch GPU. Si tenemos en cuenta el peaje del bus PCIe (el tiempo que le cuesta a PyTorch transferir la imagen gigante hacia y desde la gráfica), resulta que en la práctica, tu Numba CPU Paralelo casi iguala la experiencia completa de PyTorch GPU sin necesidad de mover los datos de la placa base.

- ¿Las salidas de PyTorch (CPU/GPU) son consistentes con Numba/NumPy en términos de bordes detectados y porcentaje de píxeles blancos, y qué factores podrían explicar diferencias?  
    Hicimos un tratamiento del marco porque el código manual (Numba/Secuencial) iteraba la matriz dejando el borde exterior de un píxel sin procesar (en negro), mientras que PyTorch, por defecto, utilizaba `padding=1` para inventar píxeles virtuales en los márgenes. Esto hacía que PyTorch procesara más área y encontrara falsos píxeles blancos adicionales en los bordes. Además, ajustamos el truncamiento numérico para poder evitar la pérdida prematura de información decimal. Originalmente, el código manual cortaba bruscamente los valores con `int()` (por ejemplo, 254.9 pasaba a ser 254 y dejaba de ser blanco). Al mantener todo en `float32` y usar `round()` al final de la ejecución, logramos igualar el redondeo estadístico de alta precisión que PyTorch utiliza internamente.

    Y realizamos una modificación de TF32 para desactivar el formato TensorFloat-32 que PyTorch habilita por defecto en las tarjetas gráficas NVIDIA modernas. Este formato sacrifica precisión decimal a cambio de acelerar los cálculos de inteligencia artificial, lo que generaba divergencias microscópicas frente a la CPU.

    Las herramientas difieren claramente en** sus propósitos de diseño por defecto (PyTorch prioriza la velocidad extrema y la tolerancia a fallos del Deep Learning, mientras que Numba/NumPy ejecutan matemática estricta y cruda), pero a fin de una demostración justa se hicieron estas modificaciones para obligar a todos los frameworks a competir bajo reglas matemáticas idénticas y medir así el rendimiento puro del hardware.

## 10. Recursos

[Explicación de Arquitectura de GPU](https://www.youtube.com/watch?v=whPSD8sdx-0)
[Tutorial de Tensores](https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html#numpy-bridge)




