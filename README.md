Entendido. Aquí tienes absolutamente **todo** unificado en un solo bloque de código. Integré la explicación del proyecto, las tablas de parámetros, los ejemplos de uso y el tutorial paso a paso de GitHub directamente dentro del `README.md`. 

De esta forma, solo tienes que copiar este bloque, pegarlo en tu archivo y tendrás una documentación de nivel profesional que incluye hasta tus propios apuntes de cómo subirlo.

```markdown
# ⏱️ Motor de Benchmarking: Hilos vs Procesos

Una herramienta de línea de comandos (CLI) en Python diseñada para ejecutar, medir y comparar el rendimiento de algoritmos utilizando diferentes enfoques de concurrencia (secuencial, multithreading y multiprocessing).

Calcula automáticamente métricas clave como **Tiempo de ejecución**, **Speed-up** y **Eficiencia**, mostrando los resultados en tiempo real y permitiendo su exportación a CSV para análisis posteriores.

---

## 🚀 Características Principales

- **Ejecución Dinámica:** Carga módulos de datos, lógica y *runners* sobre la marcha sin modificar el código fuente.
- **Métricas en Tiempo Real:** Muestra el progreso y los tiempos a medida que terminan los procesos.
- **Detección de Hardware:** Identifica automáticamente la cantidad de cores físicos y lógicos de tu sistema.
- **Ordenamiento Flexible:** Ordena la tabla final de resultados por múltiples criterios (tiempo, speedup, workers, etc.).
- **Exportación:** Genera reportes en formato CSV limpios y estructurados.

---

## 📦 Estructura de Directorios Esperada

El script espera que los archivos de lógica y datos estén organizados en carpetas (módulos), lo que permite probar diferentes Trabajos Prácticos (TPs) sin mezclar código. 

```text
/
├── benchmark.py
└── tp2/
    ├── arreglo.py      # Módulo de generación de datos
    ├── logica.py       # Módulo con la lógica a ejecutar
    ├── secuencial.py   # Runner (Ejecución de un solo hilo)
    ├── threads.py      # Runner (Multithreading)
    └── process.py      # Runner (Multiprocessing)
```

---

## ⚙️ Diccionario de Argumentos

El script se ejecuta desde la terminal llamando a `benchmark.py`. El único parámetro estrictamente obligatorio es `--desde`.

| Argumento | Tipo | Valor por defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--desde` | `String` | **REQUERIDO** | Carpeta que contiene los scripts a evaluar (ej. `tp2`). |
| `--contexto` | `String` | `"Benchmarking"` | Título descriptivo del experimento. |
| `--datos` | `String` | `'arreglo'` | Nombre del módulo generador de datos (sin el `.py`). |
| `--logica` | `String` | `'logica'` | Nombre del módulo que contiene la función o algoritmo. |
| `--runners` | `Lista` | `[]` | Lista de runners a ejecutar (ej. `secuencial threads process`). |
| `--workers` | `Lista` | `1 2 4` | Cantidad de workers (hilos/procesos) a instanciar. |
| `--cant` | `Int` | `1` | Magnitud, potencia o tamaño de datos a generar/procesar. |
| `--orden` | `Int Int` | `1 50` | Límites (min y max) para el esfuerzo de cómputo aleatorio. |
| `--input` | `Lista` | `[]` | Ingreso de datos manuales separados por espacio (ignora `--orden`). |
| `--sort` | `Lista` | `None` | Criterios para ordenar la tabla: `tiempo`, `speedup`, `eficiencia`, `tipo`, `workers`. |
| `--desc` | `Flag` | `False` | Aplica orden descendente (de mayor a menor) a la tabla. |
| `--seed` | `Int` | `2026` | Semilla para la generación de datos (garantiza reproducibilidad). |
| `--csv` | `String` | `None` | Ruta y nombre del archivo CSV para exportar los datos. |

---

## 💻 Ejemplos de Ejecución (Casos de Uso)

### 1. La prueba completa (Exportación y Ordenamiento)
Ejecuta una prueba masiva, compara enfoques, ordena los resultados para ver el mejor speed-up primero y guarda un reporte.
```bash
python benchmark.py --contexto "Prueba Matriz" --desde tp2 --datos matriz --cant 300 --runners secuencial process threads --workers 2 4 8 --sort speedup tiempo --desc --csv reporte.csv
```

### 2. Prueba rápida de control (Baseline)
Ideal para verificar que la lógica funciona correctamente antes de estresar la máquina. Corre solo el código secuencial con pocos datos.
```bash
python benchmark.py --contexto "Test de sanidad" --desde tp2 --cant 10 --runners secuencial
```

### 3. Comparativa directa: Hilos vs Procesos
Fija la cantidad de workers a 4 y compara directamente cuál de los dos paradigmas resuelve el problema más rápido.
```bash
python benchmark.py --contexto "Threads vs Process (4W)" --desde tp2 --runners threads process --workers 4 --sort tiempo
```

### 4. Ejecución con datos de entrada manuales
Si necesitas probar un caso borde específico en lugar de datos aleatorios.
```bash
python benchmark.py --contexto "Test Manual" --desde tp2 --input 15 8 99 4 0 --runners secuencial process --workers 2
```

### 5. Prueba de escalabilidad (Stress Test)
Evalúa cómo decae la eficiencia al saturar los núcleos lógicos del procesador.
```bash
python benchmark.py --contexto "Prueba de saturación" --desde tp2 --cant 1000 --runners process --workers 1 2 4 8 16 32 --sort eficiencia --desc
```