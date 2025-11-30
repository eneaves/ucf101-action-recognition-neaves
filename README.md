# UCF101 Action Recognition with Skeleton Data

Este proyecto implementa un sistema de reconocimiento de acciones humanas utilizando el dataset **UCF101** y representaciones de esqueletos (pose estimation). Se comparan dos arquitecturas de Deep Learning: un **Baseline MLP** y un modelo **LSTM**.

## Descripcion

El objetivo es clasificar videos en 101 categorías de acciones (ej. "ApplyEyeMakeup", "Biking", etc.) utilizando las coordenadas (x, y) de 17 puntos clave del cuerpo humano extraídos de cada frame.

### Modelos Implementados
1.  **Baseline MLP (`BaselineSkeletonMLP`):**
    *   Promedia los keypoints a lo largo del tiempo para obtener una representación estática.
    *   Utiliza una red neuronal totalmente conectada (MLP).
    *   Rápido de entrenar, pero ignora la secuencia temporal.

2.  **LSTM (`SkeletonLSTMModel`):**
    *   Procesa la secuencia de keypoints frame a frame.
    *   Utiliza una red recurrente (LSTM) para capturar dependencias temporales.
    *   Generalmente ofrece mejor rendimiento en acciones complejas.

## Instalacion

1.  Clona este repositorio:
    ```bash
    git clone <url-del-repo>
    cd ucf101-action-recognition
    ```

2.  Crea un entorno virtual (opcional pero recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Preparacion de Datos

**Nota:** El archivo del dataset (`ucf101_2d.pkl`) **no se incluye en este repositorio** debido a su gran tamaño (>1GB).

El proyecto espera encontrar el archivo de anotaciones en la siguiente ruta:

```
data/raw/ucf101_2d.pkl
```

Asegúrate de descargar el archivo y colocarlo en esa ubicación antes de ejecutar el código.

## Uso

### 1. Analisis y Comparacion (Recomendado)
Para entrenar ambos modelos, ver las gráficas de pérdida/accuracy y comparar resultados, ejecuta el notebook:

```bash
jupyter notebook analysis.ipynb
```

### 2. Entrenamiento en Terminal
Puedes entrenar los modelos individualmente usando los scripts en `src/`:

**Entrenar Baseline MLP:**
```bash
python src/train.py --epochs 30 --batch_size 64
```

**Entrenar LSTM:**
```bash
python src/train_lstm.py --epochs 30 --batch_size 64
```

### 3. Prediccion
Para realizar predicciones con un modelo entrenado:

```bash
python src/predict.py --model_type lstm --weights skeleton_lstm_model.pth --pkl_path data/raw/ucf101_2d.pkl
```

Si deseas probar con un video específico (por índice):
```bash
python src/predict.py --model_type lstm --weights skeleton_lstm_model.pth --pkl_path data/raw/ucf101_2d.pkl --index 100
```

## Estructura del Proyecto

```
.
├── analysis.ipynb          # Notebook para entrenamiento y comparación
├── baseline_skeleton_mlp.pth # Pesos guardados del modelo Baseline
├── skeleton_lstm_model.pth   # Pesos guardados del modelo LSTM
├── data/
│   └── raw/                # Datos de entrada (.pkl)
├── src/
│   ├── dataset_skeleton.py # Clase Dataset de PyTorch
│   ├── models/             # Definición de arquitecturas
│   │   ├── baseline_skeleton.py
│   │   └── skeleton_lstm.py
│   ├── train.py            # Script de entrenamiento para Baseline
│   ├── train_lstm.py       # Script de entrenamiento para LSTM
│   └── predict.py          # Script de inferencia
└── requirements.txt        # Dependencias
```

## Resultados y Analisis

Los resultados detallados y las curvas de aprendizaje se pueden visualizar ejecutando `analysis.ipynb`.

### Comparacion de Modelos (200 Epocas)

Tras entrenar ambos modelos durante 200 épocas con datos normalizados, observamos tendencias claras que definen las fortalezas y debilidades de cada arquitectura:

1.  **Baseline MLP (Perceptron Multicapa):**
    *   **Rendimiento:** Alcanza rápidamente una accuracy de validación alta (~36-38%), pero se estanca ahí.
    *   **Overfitting:** Muestra un sobreajuste severo. Mientras la accuracy de entrenamiento sigue subiendo (llegando a >65%), la pérdida de validación (Val Loss) empieza a oscilar violentamente y a subir, indicando que el modelo está memorizando el ruido de los datos de entrenamiento en lugar de generalizar.
    *   **Conclusion:** Es un modelo rápido y efectivo para capturar posturas estáticas promedio, pero su capacidad de aprendizaje está limitada y es propenso a memorizar.

2.  **LSTM Bidireccional (Red Recurrente):**
    *   **Rendimiento:** Comienza lento, pero mantiene una tendencia de mejora constante y estable a lo largo de todo el entrenamiento.
    *   **Estabilidad:** A diferencia del Baseline, la curva de pérdida de validación del LSTM disminuye suavemente sin los picos violentos del MLP.
    *   **Generalizacion:** La brecha entre entrenamiento y validación es menor, lo que sugiere que el modelo está aprendiendo patrones temporales reales y generalizables.
    *   **Conclusion:** Aunque requiere más tiempo de entrenamiento para converger, el LSTM demuestra ser una arquitectura superior para este problema a largo plazo, ya que es capaz de modelar la secuencia temporal de las acciones, algo que el Baseline ignora por completo.

### Mejoras Implementadas
Para lograr estos resultados, se aplicaron las siguientes mejoras técnicas respecto a la implementación inicial:
*   **Normalizacion de Datos:** Se escalaron las coordenadas de los esqueletos al rango [-1, 1], lo cual fue crucial para estabilizar el entrenamiento del LSTM.
*   **Arquitectura Bidireccional:** Se modificó el LSTM para ser bidireccional, permitiéndole tener contexto tanto del pasado como del futuro en cada frame.
*   **Aumento de Epocas:** Se extendió el entrenamiento a 200 épocas para permitir que el LSTM convergiera adecuadamente.
