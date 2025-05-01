import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Importar clases necesarias (asegúrate de que las rutas sean correctas)
from model.tsm_gru import TSM_GRU
from train_tsm_gru import HandWashDataset # Reutilizar la clase Dataset

# Dimensión de características por frame (17 keypoints * 3 valores = 51)
FEATURE_DIM = 51

# Mapeo de etiquetas a nombres (para el reporte)
LABEL_NAMES = {0: 'Mojar', 1: 'Enjabonado', 2: 'Frotado', 3: 'Aclarado'}

def evaluate_model(model, dataloader, device):
    """Evalúa el modelo en el dataloader proporcionado."""
    model.eval() # Poner el modelo en modo evaluación
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Ignorar muestras con etiqueta -1 (error de carga)
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            inputs = inputs[valid_mask]
            labels = labels[valid_mask]

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

def plot_confusion_matrix(cm, class_names, output_path="confusion_matrix_eval.png"): # Renamed default output
    """Genera y guarda un gráfico de la matriz de confusión."""
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 7))
    try:
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    except ValueError:
        # Fallback si hay problemas con los datos (ej. todo ceros)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Matriz de confusión guardada en {output_path}")
    plt.close()

def main(args):
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Construct absolute path for the model --- 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check if the provided path is absolute. If not, make it relative to the script dir.
    if not os.path.isabs(args.model_path):
        model_path_abs = os.path.join(script_dir, args.model_path)
    else:
        model_path_abs = args.model_path

    # --- Cargar Checkpoint --- 
    if not os.path.exists(model_path_abs):
        print(f"Error: No se encontró el archivo del modelo en '{model_path_abs}' (original path: '{args.model_path}')")
        print("Asegúrate de haber entrenado el modelo primero (ejecuta train_tsm_gru.py) y que el archivo .pth está en el mismo directorio que evaluate.py o que la ruta es correcta.")
        return

    print(f"Cargando checkpoint desde {model_path_abs}...")
    try:
        checkpoint = torch.load(model_path_abs, map_location=device)
    except Exception as e:
        print(f"Error al cargar el checkpoint: {e}")
        return

    # Recrear la arquitectura del modelo usando los argumentos guardados si existen
    # O usar los argumentos pasados por línea de comandos como fallback
    model_args = checkpoint.get('args', None)
    input_dim = FEATURE_DIM  # Usar dimensión de YOLOv8 pose por defecto

    if model_args:
        print("Usando hiperparámetros guardados en el checkpoint para crear el modelo.")
        # Si el modelo fue entrenado con una dimensión diferente, advertir y usar la nueva
        if 'input_dim' in model_args and model_args['input_dim'] != FEATURE_DIM:
            print(f"Advertencia: El modelo fue entrenado con input_dim={model_args['input_dim']}, pero se esperaba {FEATURE_DIM}")
            print("Esto puede ocurrir si se cambió de un modelo YOLO custom a YOLOv8 pose estándar.")
            # Vamos a usar la dimensión con la que se entrenó el modelo
            input_dim = model_args['input_dim']
            
        is_bidirectional = model_args.get('bidirectional', args.bidirectional) # Fallback to CLI arg if not saved
        use_attention = model_args.get('use_attention', args.use_attention) # Fallback to CLI arg if not saved

        model = TSM_GRU(input_dim=input_dim,
                        hidden_dim=model_args.get('hidden_dim', args.hidden_dim),
                        num_classes=model_args.get('num_classes', args.num_classes),
                        num_layers_gru=model_args.get('gru_layers', args.gru_layers),
                        tsm_segments=model_args.get('tsm_segments', args.tsm_segments),
                        tsm_shift_div=model_args.get('tsm_shift_div', args.tsm_shift_div),
                        bidirectional=is_bidirectional, # ---> Use the flag < ---
                        use_attention=use_attention,   # ---> Use the flag < ---
                        dropout=0.0 # Usar dropout 0 para evaluación
                       ).to(device)
    else:
        print("Advertencia: No se encontraron argumentos guardados en el checkpoint. Usando argumentos de línea de comandos para crear el modelo.")
        model = TSM_GRU(input_dim=FEATURE_DIM,
                        hidden_dim=args.hidden_dim,
                        num_classes=args.num_classes,
                        num_layers_gru=args.gru_layers,
                        tsm_segments=args.tsm_segments,
                        tsm_shift_div=args.tsm_shift_div,
                        bidirectional=args.bidirectional, # ---> Use CLI arg < ---
                        use_attention=args.use_attention,   # ---> Use CLI arg < ---
                        dropout=0.0 # Usar dropout 0 para evaluación
                       ).to(device)

    # Cargar los pesos del modelo
    try:
        # Load the state dict
        state_dict = checkpoint['model_state_dict']
        
        # Adjust keys if necessary (e.g., if saved with DataParallel)
        # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        print("Pesos del modelo cargados correctamente.")
    except KeyError:
        print("Error: El checkpoint no contiene 'model_state_dict'. Verifica el archivo .pth.")
        return
    except RuntimeError as e:
         print(f"Error al cargar state_dict: {e}")
         print("Asegúrate de que la arquitectura del modelo (incluyendo bidirectional, use_attention, hidden_dim, gru_layers) coincida con la guardada en el checkpoint.")
         print("Intenta pasar los mismos flags --bidirectional y --use-attention que usaste en el entrenamiento al script de evaluación.")
         return

    # --- Cargar Datos de Validación ---
    print("Cargando dataset de validación...")
    try:
        # Usar la misma longitud de secuencia que en el entrenamiento si está disponible
        seq_len = model_args.get('seq_len', args.seq_len) if model_args else args.seq_len
        if seq_len is None or seq_len <= 0:
             print(f"Advertencia: Longitud de secuencia inválida ({seq_len}). Usando default=150.")
             seq_len = 150
        
        val_dataset = HandWashDataset(args.val_dir, args.val_labels, sequence_length=seq_len)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo de datos o etiquetas de validación: {e}")
        print(f"Verifica las rutas: {args.val_dir}, {args.val_labels}")
        return
    except Exception as e:
        print(f"Error al crear el dataset de validación: {e}")
        return

    if len(val_dataset) == 0:
        print("Error: El dataset de validación está vacío.")
        return

    # Usar un batch size mayor para evaluación si hay memoria suficiente
    eval_batch_size = args.batch_size * 2
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset de validación cargado con {len(val_dataset)} muestras.")

    # --- Realizar Evaluación ---
    print("Evaluando el modelo...")
    true_labels, predictions = evaluate_model(model, val_loader, device)

    if not true_labels or not predictions:
        print("No se pudieron obtener predicciones. Verifica el dataset y el modelo.")
        return

    # --- Mostrar Resultados ---
    print("\n--- Resultados de la Evaluación ---")

    # Obtener nombres de clases para el reporte
    # Asegurarse de que las clases presentes en true_labels y predictions existan en LABEL_NAMES
    present_labels = sorted(list(set(true_labels) | set(predictions)))
    target_names = [LABEL_NAMES.get(lbl, f'Clase {lbl}') for lbl in present_labels]
    labels_for_report = present_labels # Usar los índices numéricos para el reporte

    # Matriz de Confusión
    print("\nMatriz de Confusión:")
    cm = confusion_matrix(true_labels, predictions, labels=labels_for_report)
    print(cm)
    # Graficar matriz de confusión
    plot_confusion_matrix(cm, class_names=target_names, output_path=args.cm_path)

    # Reporte de Clasificación
    print("\nReporte de Clasificación:")
    # zero_division=0 para evitar warnings si no hay soporte para una clase
    report = classification_report(true_labels, predictions, labels=labels_for_report, target_names=target_names, zero_division=0)
    print(report)

    print("\nEvaluación completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evalúa un modelo TSM-GRU entrenado.')

    # Default paths are now relative to this script's location
    parser.add_argument('--model-path', type=str, default='best_tsm_gru.pth', help='Ruta al checkpoint del modelo entrenado (.pth).')
    parser.add_argument('--val-dir', type=str, default='data/keypoints_val', help='Directorio con keypoints de validación (.npy).')
    parser.add_argument('--val-labels', type=str, default='data/keypoints_val/labels.csv', help='Archivo CSV de etiquetas de validación.')
    parser.add_argument('--cm-path', type=str, default='confusion_matrix_eval.png', help='Ruta para guardar la imagen de la matriz de confusión.') # Updated default

    # Hiperparámetros (usados como fallback si no están en el checkpoint)
    parser.add_argument('--seq-len', type=int, default=150, help='Longitud de secuencia usada en entrenamiento (fallback).')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch para evaluación (puede ser mayor que en entrenamiento).')
    parser.add_argument('--num-workers', type=int, default=0, help='Número de workers para DataLoader.')

    # Hiperparámetros del Modelo (fallback)
    parser.add_argument('--input-dim', type=int, default=FEATURE_DIM, help='Dimensión de entrada (fallback).')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Dimensión oculta GRU (fallback). Ajusta si es necesario.') # Adjusted default to match training likely
    parser.add_argument('--num-classes', type=int, default=4, help='Número de clases (fallback).')
    parser.add_argument('--gru-layers', type=int, default=2, help='Número de capas GRU (fallback). Ajusta si es necesario.') # Adjusted default to match training likely
    parser.add_argument('--tsm-segments', type=int, default=8, help='Segmentos TSM (fallback).')
    parser.add_argument('--tsm-shift-div', type=int, default=3, help='Divisor shift TSM (fallback).') # Changed default from 8 to 3
    parser.add_argument('--bidirectional', action='store_true', help='Usar GRU bidireccional (fallback/override).')
    parser.add_argument('--use-attention', action='store_true', help='Usar mecanismo de atención (fallback/override).')

    args = parser.parse_args()

    # Validate tsm_shift_div if using fallback args
    if args.input_dim % args.tsm_shift_div != 0:
        print(f"Advertencia: --input-dim ({args.input_dim}) no es divisible por --tsm-shift-div ({args.tsm_shift_div}). Esto puede causar problemas si el checkpoint no contiene argumentos.")

    main(args)
