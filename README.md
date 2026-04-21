# ⏱️ Motor de Benchmarking: Hilos vs Procesos

Una herramienta de línea de comandos (CLI) en Python diseñada para ejecutar, medir y comparar el rendimiento de algoritmos utilizando diferentes enfoques de concurrencia (secuencial, multithreading y multiprocessing).

Calcula automáticamente métricas clave como **Tiempo de ejecución**, **Speed-up** y **Eficiencia**, mostrando los resultados en tiempo real y permitiendo su exportación a CSV.

## 🚀 Características Principales

- **Ejecución Dinámica:** Carga módulos de datos, lógica y *runners* sobre la marcha.
- **Métricas en Tiempo Real:** Muestra el progreso y los tiempos a medida que terminan los procesos.
- **Detección de Hardware:** Identifica automáticamente la cantidad de cores físicos y lógicos de tu sistema.
- **Ordenamiento Flexible:** Ordena la tabla final de resultados por múltiples criterios (tiempo, speedup, workers, etc.).
- **Exportación:** Genera reportes en formato CSV para análisis posteriores.

## 🛠️ Requisitos

- Python 3.6 o superior.
- No requiere dependencias externas (utiliza la biblioteca estándar de Python).

## 📦 Estructura de Directorios Esperada

El script espera que los archivos de lógica y datos estén organizados en carpetas (módulos), por ejemplo:

```text
/
├── benchmark.py
└── tp2/
    ├── arreglo.py      # Módulo de generación de datos
    ├── logica.py       # Módulo con la lógica a ejecutar
    ├── secuencial.py   # Runner
    ├── threads.py      # Runner
    └── process.py      # Runner