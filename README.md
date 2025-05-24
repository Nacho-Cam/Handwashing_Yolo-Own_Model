# Sistema de Supervisión Automática de Lavado de Manos

Este proyecto implementa un sistema completo para la supervisión automática de lavado de manos usando visión artificial y aprendizaje profundo. Utiliza un modelo YOLO Pose afinado para la detección de keypoints de manos y un modelo TSM-GRU (Temporal Shift Module + GRU) para clasificar las fases del lavado.

![Hand Washing Classification](https://www.cdc.gov/handwashing/images/wash-your-hands-steps-1280x720.jpg)

## Estructura del Proyecto

```
.
├── lavadodemanosyoloposev11yentrenamiento/
│   ├── data/
│   │   ├── raw_videos/                  # Videos originales del dataset
│   │   ├── keypoints_train_finetuned/   # Keypoints (21) para entrenamiento (.npy)
│   │   └── keypoints_val_finetuned/     # Keypoints (21) para validación (.npy)
│   ├── model/
│   │   ├── temporal_shift.py            # Implementación del módulo Temporal Shift
│   │   └── tsm_gru.py                   # Modelo TSM-GRU para clasificación de fases
│   ├── scripts/
│   │   ├── extract_keypoints_finetuned.py # Script para extraer keypoints de manos (21)
│   │   └── compare_models.py            # Script para comparar modelos (si aplica)
│   ├── use/
│   │   └── train_handwash_pipeline.py   # Script para ejecutar el pipeline completo
│   ├── evaluate.py                      # Script para evaluar el modelo entrenado
│   ├── hand-keypoints.yaml              # Configuración del dataset para afinado de YOLO
│   ├── train_tsm_gru.py                 # Script para entrenar el modelo TSM-GRU
│   ├── yolo11n-pose-hands.pt            # Modelo YOLO afinado para manos (21 keypoints)
│   ├── yolo11n-pose.pt                  # Modelo YOLO base (17 keypoints)
│   └── best_tsm_gru.pth                 # Mejor modelo TSM-GRU entrenado
├── webcam_handwash_monitor.py           # Script para inferencia en tiempo real con webcam
├── inference.py                         # Script para inferencia en tiempo real solo con YOLO
├── requirements.txt                     # Dependencias del proyecto
├── LICENSE
└── README.md
```

## Instalación

1.  **Clonar el repositorio** (si aún no lo has hecho).
2.  **Instalar dependencias** (desde la raíz del proyecto):
    ```bash
    pip install -r requirements.txt
    ```

    Dependencias clave:
    ```
    torch>=2.0.0
    torchvision>=0.15.0
    ultralytics>=8.0.0
    opencv-python>=4.7.0
    numpy>=1.24.0
    pandas>=2.0.0
    matplotlib>=3.7.0
    seaborn>=0.12.0
    scikit-learn>=1.2.0
    tqdm>=4.65.0
    # paho-mqtt (si se usa MQTT en webcam_handwash_monitor.py)
    ```

    > **Nota**: Para versiones específicas de PyTorch con soporte CUDA, instala PyTorch por separado siguiendo las instrucciones en [pytorch.org](https://pytorch.org/).

> **Nota importante (mayo 2025):**
> Actualmente el modelo TSM-GRU **no está funcionando** en la inferencia en tiempo real. Por ahora, la evaluación de la calidad del lavado de manos se realiza únicamente usando el modelo YOLO pose, midiendo el tiempo de presencia simultánea de ambas manos en pantalla. Si ambas manos están presentes al menos 20 segundos, el lavado se considera correcto; de lo contrario, se genera una alerta.

## Inferencia en Tiempo Real Basada Solo en YOLO (sin TSM-GRU)

A partir de mayo 2025, el script `inference.py` permite realizar inferencia en tiempo real para supervisión de lavado de manos **sin necesidad de usar el modelo TSM-GRU**. Ahora, la calidad del lavado se evalúa únicamente en función del tiempo que ambas manos están presentes en pantalla.

**¿Cómo funciona?**
- Se utiliza únicamente el modelo YOLO pose para detectar ambas manos.
- Si ambas manos están presentes en pantalla durante al menos 20 segundos, se considera un lavado correcto ("OK").
- Si el tiempo es menor, se publica una alerta ("ALERTA_DURACION_INSUFICIENTE") vía MQTT.
- El temporizador y el estado se muestran en la ventana de video.

### Ejecución

Desde la raíz del proyecto:
```bash
python lavadodemanosyoloposev11yentrenamiento/inference.py --show-video
```

Parámetros opcionales:
- `--yolo-model`: Ruta al modelo YOLO pose afinado para manos (por defecto: `yolo11n-pose-hands.pt`)
- `--webcam-index`: Índice de la cámara a usar (por defecto: 0)
- `--mqtt-broker` y `--mqtt-port`: Configuración del broker MQTT

**Nota:**
- Ya no es necesario el archivo `best_tsm_gru.pth` ni la extracción de keypoints para la inferencia en tiempo real básica.
- El modelo YOLO pose debe estar entrenado o afinado para detectar ambas manos y sus keypoints.

---

## Pipeline de Entrenamiento Automatizado

El script `train_handwash_pipeline.py` automatiza la mayoría de los pasos. Se recomienda usar este script para un flujo de trabajo consistente.

(Desde la raíz del proyecto):
```bash
python lavadodemanosyoloposev11yentrenamiento/use/train_handwash_pipeline.py [OPCIONES]
```

Ejemplo para usar el modelo YOLO afinado para manos, saltar el afinado (si ya está hecho) y proceder con la extracción y entrenamiento:
```bash
python lavadodemanosyoloposev11yentrenamiento/use/train_handwash_pipeline.py --use-finetuned-model --skip-finetune
```

Consulta los argumentos del script para más opciones:
```bash
python lavadodemanosyoloposev11yentrenamiento/use/train_handwash_pipeline.py --help
```

## Implementación de Modelos

### 1. YOLO Pose (Afinado para Manos)

Este proyecto utiliza un modelo YOLO Pose afinado específicamente para la detección de keypoints de manos.

-   Modelo: `lavadodemanosyoloposev11yentrenamiento/yolo11n-pose-hands.pt`
-   Keypoints detectados: 21 keypoints por mano.
-   **Dimensión de características**: 63 (21 keypoints × 3 valores [x, y, confianza])

### 2. Modelo TSM-GRU

El modelo de clasificación temporal combina:

-   **Módulo de Desplazamiento Temporal (TSM)**: Permite la captura de información temporal al intercambiar parcialmente características entre frames adyacentes.
-   **Unidad Recurrente GRU**: Procesa secuencias largas capturando dependencias temporales.
-   Opcionalmente puede incluir mecanismos de atención y GRU bidireccional.

## Preparación de Datos (Manual)

Si no usas el pipeline, sigue estos pasos:

1.  **Descargar el dataset de videos**:
    Coloca tus videos de lavado de manos en subdirectorios dentro de `lavadodemanosyoloposev11yentrenamiento/data/raw_videos/`. Se recomienda organizar los videos en carpetas por fase (ej. `phase0`, `phase1`, `rubpalms`) ya que el script de extracción puede inferir etiquetas de estas carpetas.

2.  **Extracción de Keypoints de Manos (21 keypoints)**:
    Usa el script `extract_keypoints_finetuned.py` con el modelo `yolo11n-pose-hands.pt`.

    (Desde la raíz del proyecto):
    ```bash
    python lavadodemanosyoloposev11yentrenamiento/scripts/extract_keypoints_finetuned.py --model-path lavadodemanosyoloposev11yentrenamiento/yolo11n-pose-hands.pt --raw-dir lavadodemanosyoloposev11yentrenamiento/data/raw_videos/ --train-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_train_finetuned --val-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_val_finetuned
    ```
    Este script:
    -   Lee los videos del dataset.
    -   Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para mejorar contraste.
    -   Extrae 21 keypoints por mano usando el modelo YOLO Pose especificado.
    -   Guarda los keypoints (63 características) como archivos .npy.
    -   Crea archivos `labels.csv` con las etiquetas de clase (inferidas de nombres de archivo/directorio).

## Entrenamiento del Modelo TSM-GRU (Manual)

(Desde la raíz del proyecto):
```bash
python lavadodemanosyoloposev11yentrenamiento/train_tsm_gru.py [OPCIONES]
```

El modelo se entrenará usando la arquitectura TSM-GRU. El mejor modelo se guardará como `lavadodemanosyoloposev11yentrenamiento/best_tsm_gru.pth`.

Parámetros importantes (ajustar rutas y valores según sea necesario):
```bash
python lavadodemanosyoloposev11yentrenamiento/train_tsm_gru.py \
    --train-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_train_finetuned \
    --train-labels lavadodemanosyoloposev11yentrenamiento/data/keypoints_train_finetuned/labels.csv \
    --val-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_val_finetuned \
    --val-labels lavadodemanosyoloposev11yentrenamiento/data/keypoints_val_finetuned/labels.csv \
    --save-path lavadodemanosyoloposev11yentrenamiento/best_tsm_gru.pth \
    --input-dim 63 \
    --num-classes 4 \
    --seq-len 150 \
    --batch-size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --hidden-dim 256 \
    --gru-layers 2 \
    --tsm-segments 8 \
    --tsm-shift-div 3 \
    --dropout 0.5 \
    --bidirectional \
    --use-attention \
    --use-class-weights \
    --use-lr-scheduler
    # ... y más opciones disponibles, ver --help
```

## Evaluación del Modelo TSM-GRU (Manual)

(Desde la raíz del proyecto):
```bash
python lavadodemanosyoloposev11yentrenamiento/evaluate.py --model-path lavadodemanosyoloposev11yentrenamiento/best_tsm_gru.pth --val-dir lavadodemanosyoloposev11yentrenamiento/data/keypoints_val_finetuned --val-labels lavadodemanosyoloposev11yentrenamiento/data/keypoints_val_finetuned/labels.csv --input-dim 63 --num-classes 4 [OPCIONES_ARQUITECTURA_MODELO]
```
Esto generará:
-   Una matriz de confusión (`lavadodemanosyoloposev11yentrenamiento/confusion_matrix_eval.png`)
-   Un reporte de clasificación con precisión, recall y F1-score para cada fase.

Asegúrate de pasar los argumentos de arquitectura (`--bidirectional`, `--use-attention`, `--hidden-dim`, etc.) que coincidan con el modelo entrenado si no están guardados en el checkpoint.

## Inferencia en Tiempo Real con Webcam

Usa el script `webcam_handwash_monitor.py` para monitoreo y evaluación automática de la calidad del lavado de manos.

(Desde la raíz del proyecto):
```bash
python webcam_handwash_monitor.py
```

Características principales:
- Captura video desde webcam en tiempo real.
- Aplica CLAHE y extrae keypoints de manos usando el modelo YOLO pose afinado (`yolo11n-pose-hands.pt`).
- Clasifica fases del lavado usando TSM-GRU y muestra la fase detectada en pantalla.
- Evalúa la calidad del lavado (por duración de fases) y la muestra en la parte inferior del video.
- Overlay de información clara: fase actual, tiempo en fase, duración por fase, calidad general.
- Todos los textos en la interfaz de video están en español y son descriptivos.

Parámetros opcionales (ver el script para más detalles):
```bash
python webcam_handwash_monitor.py \
    --yolo-model lavadodemanosyoloposev11yentrenamiento/yolo11n-pose-hands.pt \
    --tsm-gru-model lavadodemanosyoloposev11yentrenamiento/best_tsm_gru.pth \
    --webcam-index 0
```

---

## Cambios recientes en la visualización
- Los textos superpuestos en el video ahora son más claros y están en español:
    - "Fase actual: ..."
    - "Tiempo en fase: ..."
    - "Duración por fase (s): ..."
    - "Calidad general: ..."
- El color de la calidad general es verde si es "Good" o "Excellent", rojo en otros casos.
- Se omite la fase "No Handwashing" en el resumen de duración por fases.

---

## Etiquetas de Clases (Ejemplo)
-   0: No Handwashing / Other
-   1: Phase 1 (e.g., Wet/Soap)
-   2: Phase 2 (e.g., Rubbing)
-   3: Phase 3 (e.g., Rinsing)
    *(Estas etiquetas son configurables en `webcam_handwash_monitor.py` y dependen de cómo se entrenó el modelo TSM-GRU)*

## Fine-tuning del Modelo YOLO Pose (Opcional pero Recomendado)

Para obtener el modelo `yolo11n-pose-hands.pt` (o similar), puedes afinar un modelo YOLO Pose base (como `yolo11n-pose.pt`) en un dataset específico de manos.

1.  **Prepara el Dataset**:
    *   Ultralytics puede descargar automáticamente datasets conocidos. Para el dataset `hand-keypoints` de Ultralytics, necesitarás un archivo YAML (p.ej., `lavadodemanosyoloposev11yentrenamiento/hand-keypoints.yaml`):
        ```yaml
        # Contenido para lavadodemanosyoloposev11yentrenamiento/hand-keypoints.yaml
        # Ruta al directorio del dataset (Ultralytics lo descargará aquí si no existe)
        # path: ../datasets/hand-keypoints # Ajusta si tu estructura es diferente
        # O usa la descarga automática:
        download: https://github.com/ultralytics/yolov5/releases/download/v1.0/hand-keypoints.zip

        train: hand-keypoints/images/train
        val: hand-keypoints/images/val

        # Metadatos del dataset de keypoints
        kpt_shape: [21, 2]  # 21 keypoints, 2 dimensiones (x, y)
        flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # Ajustar según la definición de keypoints

        # Clases (solo una clase: 'hand')
        names:
          0: hand
        ```
        *Nota: Verifica la documentación de Ultralytics y la estructura del dataset `hand-keypoints` para los valores exactos de `kpt_shape` y `flip_idx`.*

2.  **Ejecuta el Fine-tuning**:
    *   Puedes usar la línea de comandos de Ultralytics. Desde la raíz del proyecto:
        ```bash
        yolo pose train data=lavadodemanosyoloposev11yentrenamiento/hand-keypoints.yaml model=lavadodemanosyoloposev11yentrenamiento/yolo11n-pose.pt epochs=100 imgsz=640 project=lavadodemanosyoloposev11yentrenamiento/runs/pose name=finetune_hands
        ```
    *   Ajusta `epochs`, `imgsz` y otros hiperparámetros según necesites.

3.  **Usa el Modelo Afinado**:
    *   El modelo afinado se guardará (por defecto) en `lavadodemanosyoloposev11yentrenamiento/runs/pose/finetune_hands/weights/best.pt`.
    *   Copia este archivo a `lavadodemanosyoloposev11yentrenamiento/yolo11n-pose-hands.pt` (o el nombre que prefieras) para usarlo en los scripts.
    *   Este modelo afinado detectará 21 keypoints por mano, resultando en `INPUT_DIM = 63` para el modelo TSM-GRU.

## Despliegue con Docker

(Desde la raíz del proyecto):
```bash
# Construir la imagen
docker build -t handwash-monitor -f lavadodemanosyoloposev11yentrenamiento/docker/Dockerfile .

# Ejecutar el contenedor (ejemplo para Linux con webcam en /dev/video0)
# Ajusta --device según tu sistema operativo si es necesario
# Para Windows, el acceso a la webcam desde Docker puede ser complejo.
docker run --rm -it --device=/dev/video0:/dev/video0 handwash-monitor
```
Asegúrate de que el `Dockerfile` y los scripts referenciados (como `inference.py` o `webcam_handwash_monitor.py`) usen las rutas correctas a los modelos dentro del contenedor. El `Dockerfile` actual usa `inference.py` y `yolo11n-pose.pt`. Deberás ajustarlo si `webcam_handwash_monitor.py` y `yolo11n-pose-hands.pt` son los predeterminados.

## Créditos

Este proyecto utiliza el Hand Wash Dataset (o similar), diseñado para reconocer las diferentes fases del lavado de manos siguiendo las pautas de la OMS.