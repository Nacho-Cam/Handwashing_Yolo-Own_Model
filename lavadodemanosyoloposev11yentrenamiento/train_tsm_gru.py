import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model.tsm_gru import TSM_GRU # Importar el modelo
from tqdm import tqdm
import argparse
from collections import Counter
import random
import json
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Dimensión de características por frame (21 keypoints * 3 valores = 63)
FEATURE_DIM = 63

# --- Set random seed for reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Early stopping utility ---
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_f1_max = -float('inf')
        self.delta = delta

    def __call__(self, val_f1, model, path):
        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)
        if self.verbose:
            print(f'Saved model checkpoint to {path}')

# --- Dataset Class ---
class HandWashDataset(Dataset):
    """Dataset para cargar secuencias de keypoints y etiquetas."""
    def __init__(self, keypoints_dir, labels_file, sequence_length=None):
        """
        Args:
            keypoints_dir (str): Directorio que contiene los archivos .npy.
            labels_file (str): Ruta al archivo labels.csv.
            sequence_length (int, optional): Longitud fija a la que se truncarán/rellenarán
                                             las secuencias. Si es None, usa la longitud original.
        """
        self.keypoints_dir = keypoints_dir
        self.labels_df = pd.read_csv(labels_file)
        self.sequence_length = sequence_length

        # Validar que todos los archivos .npy listados en labels.csv existen
        missing_files = []
        for npy_filename in self.labels_df['filename']:
            filepath = os.path.join(self.keypoints_dir, npy_filename)
            if not os.path.exists(filepath):
                missing_files.append(npy_filename)
        if missing_files:
            print(f"Advertencia: Faltan {len(missing_files)} archivos .npy listados en {labels_file}:")
            self.labels_df = self.labels_df[~self.labels_df['filename'].isin(missing_files)]
            print(f"Se usarán {len(self.labels_df)} muestras válidas.")

        # Calcular la distribución de clases
        label_counts = self.labels_df['label'].value_counts()
        print("Distribución de clases:")
        for label, count in label_counts.items():
            print(f"Clase {label}: {count} muestras ({count/len(self.labels_df)*100:.1f}%)")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npy_filename = self.labels_df.iloc[idx]['filename']
        label = int(self.labels_df.iloc[idx]['label'])
        filepath = os.path.join(self.keypoints_dir, npy_filename)

        try:
            keypoints = np.load(filepath) # Carga como (T, 63) para YOLOv8 pose (21 keypoints)
        except Exception as e:
            print(f"Error al cargar {filepath}: {e}. Devolviendo tensor nulo.")
            keypoints = np.zeros((self.sequence_length or 1, FEATURE_DIM), dtype=np.float32)
            label = -1

        # Verificar que el archivo .npy tenga la dimensión correcta
        if keypoints.shape[1] != FEATURE_DIM:
            print(f"Advertencia: Dimensión incorrecta en {filepath}. Esperada {FEATURE_DIM}, pero es {keypoints.shape[1]}.")
            keypoints = np.zeros((self.sequence_length or keypoints.shape[0], FEATURE_DIM), dtype=np.float32)
            label = -1

        # Truncar o rellenar la secuencia si sequence_length está definido
        if self.sequence_length:
            current_len = keypoints.shape[0]
            if current_len > self.sequence_length:
                keypoints = keypoints[-self.sequence_length:, :]
            elif current_len < self.sequence_length:
                padding = np.zeros((self.sequence_length - current_len, keypoints.shape[1]), dtype=keypoints.dtype)
                keypoints = np.vstack((padding, keypoints))

        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return keypoints_tensor, label_tensor

# --- Augmentations for Time Series ---
def apply_time_series_augmentation(keypoints_batch, labels_batch, args):
    """
    Apply various augmentations to the keypoints sequences.
    
    Args:
        keypoints_batch (tensor): Batch of keypoint sequences, shape (B, T, C)
        labels_batch (tensor): Batch of labels, shape (B,)
        args: Command line arguments with augmentation flags
    
    Returns:
        augmented_keypoints (tensor): Augmented batch
        augmented_labels (tensor): Possibly modified labels
    """
    device = keypoints_batch.device
    batch_size = keypoints_batch.shape[0]
    
    # Apply mixup if enabled
    if args.mixup and batch_size > 1:
        return apply_mixup(keypoints_batch, labels_batch, alpha=args.mixup_alpha, num_classes=args.num_classes)
    
    # Apply other augmentations if enabled
    if args.augment:
        augmented_keypoints = keypoints_batch.clone()
        
        for i in range(batch_size):
            # Only apply to valid sequences (non-zero)
            if torch.sum(torch.abs(keypoints_batch[i])) > 0:
                # Randomly select an augmentation
                aug_type = np.random.choice(['jitter', 'scaling', 'time_warp', 'none'], 
                                          p=[0.3, 0.3, 0.3, 0.1])
                
                if aug_type == 'jitter':
                    noise = torch.randn_like(keypoints_batch[i]) * 0.05  # 5% noise
                    augmented_keypoints[i] = keypoints_batch[i] + noise
                
                elif aug_type == 'scaling':
                    # Apply random scaling between 0.9 and 1.1
                    scale_factor = torch.FloatTensor(1).uniform_(0.9, 1.1).to(device)
                    augmented_keypoints[i] = keypoints_batch[i] * scale_factor
                
                elif aug_type == 'time_warp':
                    # Temporal warping - stretch or compress parts of the sequence
                    seq_len = keypoints_batch.shape[1]
                    if seq_len > 10:  # Only apply to longer sequences
                        # Create indices for temporal stretching
                        src_idx = torch.arange(seq_len, device=device).float()
                        
                        # Random warping intensity
                        warp_intensity = torch.FloatTensor(1).uniform_(0.9, 1.1).to(device)
                        
                        # Create random warping function
                        warp_center = torch.randint(low=seq_len//4, high=3*seq_len//4, size=(1,)).to(device)
                        warp_dist = torch.abs(src_idx - warp_center)
                        warp_factor = 1 + (warp_intensity - 1) * torch.exp(-warp_dist / (seq_len / 4))
                        
                        # Apply warping to indices
                        dst_idx = torch.cumsum(warp_factor, dim=0) - warp_factor[0]
                        dst_idx = dst_idx / dst_idx[-1] * (seq_len - 1)
                        
                        # Ensure monotonicity
                        dst_idx, _ = torch.sort(dst_idx)
                        
                        # Use grid_sample for smooth interpolation (1D as 2D)
                        src_keypoints = keypoints_batch[i].unsqueeze(0).permute(0, 2, 1).unsqueeze(-1)  # (1, C, T, 1)
                        grid = torch.zeros(1, seq_len, 1, 2, device=device)  # (N, H, W, 2)
                        grid[..., 0] = (2 * dst_idx / (seq_len - 1) - 1).unsqueeze(0).unsqueeze(-1)  # x-coord (temporal)
                        grid[..., 1] = 0  # y-coord (dummy, always 0)
                        warped = torch.nn.functional.grid_sample(
                            src_keypoints, grid, mode='bilinear', align_corners=True
                        )
                        
                        augmented_keypoints[i] = warped.squeeze(0).squeeze(-1).permute(1, 0)
        
        return augmented_keypoints, labels_batch
    
    # Return original data if no augmentation applied
    return keypoints_batch, labels_batch

def apply_mixup(x, y, alpha=0.2, num_classes=None):
    """Apply mixup augmentation to the batch."""
    batch_size = x.shape[0]
    device = x.device
    if num_classes is None:
        num_classes = y.max().item() + 1
    y_onehot = torch.zeros(batch_size, num_classes, device=device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]
    return mixed_x, mixed_y

# --- Funciones de Entrenamiento y Validación ---
def train_epoch(model, dataloader, criterion, optimizer, device, args, writer=None, epoch=0):
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    batch_count = 0

    pbar = tqdm(dataloader, desc="Entrenamiento")
    for inputs, labels in pbar:
        batch_count += 1

        inputs, labels = inputs.to(device), labels.to(device)

        valid_mask = labels != -1
        if not valid_mask.any():
            continue
        inputs = inputs[valid_mask]
        labels = labels[valid_mask]

        # Apply augmentations if enabled
        if args.augment or args.mixup:
            inputs, augmented_labels = apply_time_series_augmentation(inputs, labels, args)
            
            # If we got soft labels from mixup
            if args.mixup and augmented_labels.dim() > 1:
                optimizer.zero_grad()
                outputs = model(inputs)
                # Use soft targets
                loss = torch.sum(-augmented_labels * torch.log_softmax(outputs, dim=1), dim=1).mean()
            else:
                # Normal training with hard labels
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            # Normal training without augmentation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        # For metrics, always use hard labels
        _, predicted = torch.max(outputs.data, 1)
        if args.mixup and 'augmented_labels' in locals() and augmented_labels.dim() > 1:
            # For mixup, use the original labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        else:
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': running_loss / total_samples})

        # TensorBoard logging
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_count
            writer.add_scalar('Batch/train_loss', loss.item(), global_step)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    if writer is not None:
        writer.add_scalar('Epoch/train_loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/train_f1', train_f1, epoch)

    return epoch_loss, train_f1

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Validación")
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            valid_mask = labels != -1
            if not valid_mask.any():
                continue
            inputs = inputs[valid_mask]
            labels = labels[valid_mask]

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': running_loss / total_samples})

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    if len(all_preds) > 0:
        print("\nInforme de clasificación:")
        print(classification_report(all_labels, all_preds, zero_division=0))

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicción')
        plt.ylabel('Verdad')
        plt.title('Matriz de Confusión')
        plt.savefig('confusion_matrix_train.png') # Renamed to avoid conflict with evaluate
        print(f"Matriz de confusión guardada como 'confusion_matrix_train.png'")

    return epoch_loss, f1

# --- Script Principal ---
def main(args):
    # Set random seed
    set_seed(getattr(args, 'seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("Cargando datasets...")
    try:
        train_dataset = HandWashDataset(args.train_dir, args.train_labels, args.seq_len)
        val_dataset = HandWashDataset(args.val_dir, args.val_labels, args.seq_len)
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo de datos o etiquetas: {e}")
        print("Asegúrate de haber ejecutado 'scripts/extract_keypoints.py' primero.")
        return
    except Exception as e:
        print(f"Error al crear los datasets: {e}")
        return

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Uno o ambos datasets están vacíos. Verifica los archivos .npy y labels.csv.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Tamaño dataset entrenamiento: {len(train_dataset)}, validación: {len(val_dataset)}")
    
    # Calcular pesos por clase para manejar el desbalance
    if args.use_class_weights:
        print("Calculando pesos por clase para manejar el desbalance...")
        # Contar las etiquetas en el dataset de entrenamiento
        labels_count = [0] * args.num_classes
        for _, label in train_dataset:
            if label.item() >= 0 and label.item() < args.num_classes:  # Ignorar etiquetas inválidas
                labels_count[label.item()] += 1
        
        # Calcular pesos inversos al número de muestras por clase
        total_samples = sum(labels_count)
        class_weights = [total_samples / (c * args.num_classes) if c > 0 else 1.0 for c in labels_count]
        print(f"Conteo de clases: {labels_count}")
        print(f"Pesos de clases: {class_weights}")
        class_weights = torch.FloatTensor(class_weights).to(device)
    else:
        class_weights = None

    print("Creando modelo...")
    model = TSM_GRU(input_dim=args.input_dim,
                   hidden_dim=args.hidden_dim,
                   num_classes=args.num_classes,
                   num_layers_gru=args.gru_layers,
                   tsm_segments=args.tsm_segments,
                   tsm_shift_div=args.tsm_shift_div,
                   dropout=args.dropout,
                   bidirectional=args.bidirectional,
                   use_attention=args.use_attention,
                   use_batch_norm=args.use_batch_norm).to(device)

    # Choose loss function based on arguments
    if args.use_focal_loss:
        from model.tsm_gru import FocalLoss
        print(f"Usando Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma}) para manejar desbalance de clases")
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

    # TensorBoard writer
    writer = None
    if getattr(args, 'tensorboard', False) and TENSORBOARD_AVAILABLE:
        log_dir = os.path.join('runs', f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging enabled at {log_dir}")

    # Resume from checkpoint
    start_epoch = 0
    best_val_f1 = -1.0
    if getattr(args, 'resume', None):
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_f1 = checkpoint.get('f1', -1.0)
        else:
            print(f"Checkpoint {args.resume} not found. Starting from scratch.")

    # Early stopping
    early_stopping = EarlyStopping(patience=getattr(args, 'early_stopping_patience', 15), verbose=True)

    print("Iniciando entrenamiento...")
    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []

    best_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device, args, writer, epoch)
        val_loss, val_f1 = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        if writer is not None:
            writer.add_scalar('Epoch/val_loss', val_loss, epoch)
            writer.add_scalar('Epoch/val_f1', val_f1, epoch)

        if args.use_lr_scheduler:
            scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'f1': val_f1,
                'args': vars(args)
            }, args.save_path)
            print(f"Mejor modelo guardado en {args.save_path} (F1: {val_f1:.4f})")

        # Early stopping check
        early_stopping(val_f1, model, args.save_path)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    print("\nEntrenamiento completado.")
    print(f"Mejor F1 de validación obtenido: {best_val_f1:.4f} en la época {best_epoch}")
    print(f"El mejor modelo se guardó en: {args.save_path}")

    # Save config and best epoch
    config_save_path = os.path.join(os.path.dirname(args.save_path), f"train_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(config_save_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_save_path}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label='Train')
    plt.plot(val_f1s, label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Gráficos de métricas guardados como 'training_metrics.png'")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrena un modelo TSM-GRU para clasificación de lavado de manos.')

    # Paths relative to the script's directory (lavadodemanosyoloposev11yentrenamiento)
    parser.add_argument('--train-dir', type=str, default='C:\\Users\\nycca\\OneDrive\\Documentos\\NACHO\\Universidad\\Curso 3\\sistemasDePercepcionArtificialYVisionArtificial\\Ejercicio_1\\Proyecto final\\Handwashing_Yolo-Own_Model\\lavadodemanosyoloposev11yentrenamiento\\data\\keypoints_train_finetuned', help='Directorio con keypoints de entrenamiento (.npy).')
    parser.add_argument('--train-labels', type=str, default='C:\\Users\\nycca\\OneDrive\\Documentos\\NACHO\\Universidad\\Curso 3\\sistemasDePercepcionArtificialYVisionArtificial\\Ejercicio_1\\Proyecto final\\Handwashing_Yolo-Own_Model\\lavadodemanosyoloposev11yentrenamiento\\data\\keypoints_train_finetuned\\labels.csv', help='Archivo CSV de etiquetas de entrenamiento.')
    parser.add_argument('--val-dir', type=str, default='C:\\Users\\nycca\\OneDrive\\Documentos\\NACHO\\Universidad\\Curso 3\\sistemasDePercepcionArtificialYVisionArtificial\\Ejercicio_1\\Proyecto final\\Handwashing_Yolo-Own_Model\\lavadodemanosyoloposev11yentrenamiento\\data\\keypoints_val_finetuned', help='Directorio con keypoints de validación (.npy).')
    parser.add_argument('--val-labels', type=str, default='C:\\Users\\nycca\\OneDrive\\Documentos\\NACHO\\Universidad\\Curso 3\\sistemasDePercepcionArtificialYVisionArtificial\\Ejercicio_1\\Proyecto final\\Handwashing_Yolo-Own_Model\\lavadodemanosyoloposev11yentrenamiento\\data\\keypoints_val_finetuned\\labels.csv', help='Archivo CSV de etiquetas de validación.')
    # Save path relative to the script's directory
    parser.add_argument('--save-path', type=str, default='best_tsm_gru.pth', help='Ruta para guardar el mejor checkpoint del modelo.')

    parser.add_argument('--seq-len', type=int, default=150, help='Longitud fija de secuencia (truncar/rellenar). None para usar longitud original.')
    parser.add_argument('--batch-size', type=int, default=16, help='Tamaño del batch.')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento.') # Increased epochs
    parser.add_argument('--lr', type=float, default=1e-4, help='Tasa de aprendizaje para Adam.')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay para regularización L2.') # Added weight decay
    parser.add_argument('--use-class-weights', action='store_true', help='Usar pesos de clase para balancear clases.')
    parser.add_argument('--use-lr-scheduler', action='store_true', help='Usar scheduler de tasa de aprendizaje.') # Added LR scheduler option
    parser.add_argument('--num-workers', type=int, default=0, help='Número de workers para DataLoader (0 en Windows suele ser más estable).')

    parser.add_argument('--input-dim', type=int, default=FEATURE_DIM, help='Dimensión de entrada (21 kpts * 3).')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Dimensión oculta de la GRU.') # Increased hidden dim
    parser.add_argument('--num-classes', type=int, default=7, help='Número de clases (fases de lavado).')
    parser.add_argument('--gru-layers', type=int, default=2, help='Número de capas GRU.') # Increased GRU layers
    parser.add_argument('--tsm-segments', type=int, default=8, help='Número de segmentos TSM.')
    parser.add_argument('--tsm-shift-div', type=int, default=3, help='Divisor de shift TSM (debe ser divisor de input_dim).') # Changed default from 8 to 3
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout antes de la capa final.') # Increased dropout
    parser.add_argument('--bidirectional', action='store_true', help='Usar GRU bidireccional.')
    parser.add_argument('--use-attention', action='store_true', help='Usar mecanismo de atención.')
    parser.add_argument('--use-batch-norm', action='store_true', help='Usar batch normalization para input features.')
    parser.add_argument('--use-focal-loss', action='store_true', help='Usar Focal Loss en lugar de CrossEntropy para manejar mejor el desbalance de clases.')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Parámetro alpha para Focal Loss (si se usa).')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Parámetro gamma para Focal Loss (si se usa).')
    parser.add_argument('--mixup', action='store_true', help='Usar mixup augmentation durante entrenamiento.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Parámetro alpha para mixup (si se usa).')
    parser.add_argument('--augment', action='store_true', help='Aplicar augmentación de datos (TP/FP) durante entrenamiento.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training.')
    parser.add_argument('--early-stopping-patience', type=int, default=15, help='Patience for early stopping.')

    args = parser.parse_args()

    # Validate tsm_shift_div
    if args.input_dim % args.tsm_shift_div != 0:
        parser.error(f'--input-dim ({args.input_dim}) debe ser divisible por --tsm-shift-div ({args.tsm_shift_div}).')

    if args.seq_len is None or args.seq_len <= 0:
         print("Advertencia: Se requiere una longitud de secuencia fija (--seq-len > 0) para el DataLoader estándar. Usando default=150.")
         args.seq_len = 150

    main(args)
