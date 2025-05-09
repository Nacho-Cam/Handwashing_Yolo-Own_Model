import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import os
import sys

# Attempt to import TSM_GRU model.
# This assumes that 'lavadodemanosyoloposev11yentrenamiento' is in the Python path
# or that webcam_handwash_monitor.py is run from a location where this import is resolvable.
# If 'lavadodemanosyoloposev11yentrenamiento' is a package, this should work.
# You might need to add __init__.py files in 'lavadodemanosyoloposev11yentrenamiento'
# and 'lavadodemanosyoloposev11yentrenamiento/model' directories.
try:
    from lavadodemanosyoloposev11yentrenamiento.model.tsm_gru import TSM_GRU
except ImportError:
    print("ERROR: Could not import TSM_GRU from lavadodemanosyoloposev11yentrenamiento.model.tsm_gru.")
    print("Please ensure the path is correct and __init__.py files exist if it's a package.")
    sys.exit(1)

# Add phase name mapping for 7 steps
from phases import PHASES, get_phase_name

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(SCRIPT_DIR, 'lavadodemanosyoloposev11yentrenamiento', 'yolo11n-pose-hands.pt')
TSM_GRU_MODEL_PATH = os.path.join(SCRIPT_DIR, 'lavadodemanosyoloposev11yentrenamiento', 'best_tsm_gru.pth')

# Default TSM-GRU parameters (will be overridden by checkpoint if available)
SEQUENCE_LENGTH = 150
INPUT_DIM = 63  # 21 keypoints * 3 (x, y, conf)
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_CLASSES = 4
DEFAULT_GRU_LAYERS = 2
DEFAULT_TSM_SEGMENTS = 8
DEFAULT_TSM_SHIFT_DIV = 3 
DEFAULT_DROPOUT = 0.5
DEFAULT_BIDIRECTIONAL = False
DEFAULT_USE_ATTENTION = False
DEFAULT_USE_BATCH_NORM = False

WEBCAM_INDEX = 0
YOLO_CONF_THRESHOLD = 0.3
HAND_KP_CONF_THRESHOLD = 0.3 # Minimum confidence to draw a keypoint
NUM_HAND_KEYPOINTS = 21 # For yolo11n-pose-hands.pt

# Phase labels - now for 7 classes, matching phases.py
PHASE_LABELS = {
    0: "No Handwashing",
    1: "Start/Wet Hands",
    2: "Apply Soap",
    3: "Rub Palms",
    4: "Rub Back of Hands",
    5: "Interlace Fingers",
    6: "Clean Thumbs",
    7: "Rinse Hands"
}

# Ideal time per phase in seconds (customize as needed)
IDEAL_PHASE_DURATIONS = {
    PHASE_LABELS[1]: 5,   # Start/Wet Hands
    PHASE_LABELS[2]: 5,   # Apply Soap
    PHASE_LABELS[3]: 10,  # Rub Palms
    PHASE_LABELS[4]: 10,  # Rub Back of Hands
    PHASE_LABELS[5]: 10,  # Interlace Fingers
    PHASE_LABELS[6]: 10,  # Clean Thumbs
    PHASE_LABELS[7]: 10   # Rinse Hands
}
MIN_DURATION_PERCENTAGE = 0.7 # Minimum 70% of ideal time for a phase to be "Good"

# Colors for drawing
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

# --- Hand Skeleton Definition (for 21 keypoints) ---
# Based on common hand models like MediaPipe Hands
HAND_SKELETON = [
    # Palm
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17), # Wrist to finger bases
    (1, 5), (5, 9), (9, 13), (13, 17),       # Across finger bases (MCPs)

    # Thumb
    (1, 2), (2, 3), (3, 4),

    # Index Finger
    (5, 6), (6, 7), (7, 8),

    # Middle Finger
    (9, 10), (10, 11), (11, 12),

    # Ring Finger
    (13, 14), (14, 15), (15, 16),

    # Pinky Finger
    (17, 18), (18, 19), (19, 20)
]
COLOR_SKELETON = COLOR_BLUE # Color for skeleton lines

# --- Model Loading ---
def load_yolo_model(model_path, device):
    """Loads the YOLO pose model."""
    if not os.path.exists(model_path):
        print(f"ERROR: YOLO model not found at {model_path}")
        return None
    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"YOLO model loaded from {model_path} on {device}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def load_tsm_gru_model(model_path, device):
    global PHASE_LABELS
    """Loads the TSM-GRU model and its training arguments from checkpoint. Handles both full checkpoint and raw state_dict."""
    if not os.path.exists(model_path):
        print(f"ERROR: TSM-GRU model not found at {model_path}")
        return None, {}
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # --- PATCH: Handle both checkpoint dict and raw state_dict ---
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            train_args = checkpoint.get('args', {})
            num_classes = train_args.get('num_classes', DEFAULT_NUM_CLASSES)
            model = TSM_GRU(
                input_dim=train_args.get('input_dim', INPUT_DIM),
                hidden_dim=train_args.get('hidden_dim', DEFAULT_HIDDEN_DIM),
                num_classes=num_classes,
                num_layers_gru=train_args.get('gru_layers', DEFAULT_GRU_LAYERS),
                tsm_segments=train_args.get('tsm_segments', DEFAULT_TSM_SEGMENTS),
                tsm_shift_div=train_args.get('tsm_shift_div', DEFAULT_TSM_SHIFT_DIV),
                dropout=train_args.get('dropout', DEFAULT_DROPOUT),
                bidirectional=train_args.get('bidirectional', DEFAULT_BIDIRECTIONAL),
                use_attention=train_args.get('use_attention', DEFAULT_USE_ATTENTION),
                use_batch_norm=train_args.get('use_batch_norm', DEFAULT_USE_BATCH_NORM)
            )
            model = model.to(device)
            print("TSM-GRU Model effective arguments for loading:")
            for k, v in train_args.items():
                print(f"  TSM-GRU Arg - {k}: {v}")
            try:
                original_state_dict = checkpoint['model_state_dict']
                adapted_state_dict = {}
                remapped_keys_info = []
                discarded_keys_info = []
                current_model_keys = set(model.state_dict().keys())
                for k_old, v_old in original_state_dict.items():
                    k_new = k_old
                    action = "kept"
                    if k_old.startswith("gru_block.gru."):
                        k_new = k_old.replace("gru_block.gru.", "gru.")
                        action = f"renamed to {k_new}"
                    elif k_old.startswith("fc1."):
                        k_new = k_old.replace("fc1.", "fc.")
                        action = f"renamed to {k_new}"
                    elif k_old.startswith("gru_block.") or k_old.startswith("fc2."):
                        action = "discarded (no new equivalent)"
                        discarded_keys_info.append(f"  - {k_old} (reason: {action})")
                        continue
                    if k_new != k_old:
                        remapped_keys_info.append(f"  - {k_old} -> {k_new}")
                    adapted_state_dict[k_new] = v_old
                if remapped_keys_info:
                    print("\nINFO: Remapping state_dict keys for TSM-GRU model compatibility:")
                    for info in remapped_keys_info:
                        print(info)
                if discarded_keys_info:
                    print("INFO: Discarding the following state_dict keys from old model (no new equivalent):")
                    for info in discarded_keys_info:
                        print(info)
                missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
                print(f"TSM-GRU model loaded from {model_path} on {device} (with key adaptation and strict=False).")
                if missing_keys:
                    print("WARNING: Some weights were missing in the checkpoint for the current model structure (randomly initialized):")
                    for key in missing_keys:
                        print(f"  - {key}")
                if unexpected_keys:
                    print("WARNING: Some weights from the checkpoint were not used by the current model structure (ignored):")
                    for key in unexpected_keys:
                        print(f"  - {key}")
                print("INFO: Performance may be affected due to model structure changes since the checkpoint was saved.")
                print("INFO: Retraining with the current model structure is recommended for optimal results.")
            except RuntimeError as e:
                print(f"Error loading TSM-GRU model state_dict even after remapping and strict=False: {e}")
                return None, {}
            except Exception as e:
                print(f"An unexpected error occurred during TSM-GRU model loading: {e}")
                return None, {}
            model.eval()
            print(f"TSM-GRU model loaded from {model_path} on {device}")
            if num_classes != DEFAULT_NUM_CLASSES:
                print(f"Updating PHASE_LABELS for {num_classes} classes.")
                PHASE_LABELS = {i: f"Phase {i}" for i in range(num_classes)}
                if 0 in PHASE_LABELS: PHASE_LABELS[0] = "No Action / Other"
            return model, train_args
        else:
            # Assume raw state_dict (no args, use defaults)
            print("INFO: Loaded raw state_dict (no checkpoint dictionary). Attempting to infer num_classes from state_dict.")
            # Try to infer num_classes from fc.weight or fc.bias
            num_classes = DEFAULT_NUM_CLASSES
            if 'fc.weight' in checkpoint:
                num_classes = checkpoint['fc.weight'].shape[0]
            elif 'fc.bias' in checkpoint:
                num_classes = checkpoint['fc.bias'].shape[0]
            else:
                print("WARNING: Could not infer num_classes from state_dict. Using default.")
            print(f"INFO: Inferred num_classes = {num_classes}")
            model = TSM_GRU(
                input_dim=INPUT_DIM,
                hidden_dim=DEFAULT_HIDDEN_DIM,
                num_classes=num_classes,
                num_layers_gru=DEFAULT_GRU_LAYERS,
                tsm_segments=DEFAULT_TSM_SEGMENTS,
                tsm_shift_div=DEFAULT_TSM_SHIFT_DIV,
                dropout=DEFAULT_DROPOUT,
                bidirectional=DEFAULT_BIDIRECTIONAL,
                use_attention=DEFAULT_USE_ATTENTION,
                use_batch_norm=DEFAULT_USE_BATCH_NORM
            )
            model = model.to(device)
            try:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                print(f"TSM-GRU model loaded from {model_path} on {device} (raw state_dict, strict=False).")
                if missing_keys:
                    print("WARNING: Some weights were missing in the checkpoint for the current model structure (randomly initialized):")
                    for key in missing_keys:
                        print(f"  - {key}")
                if unexpected_keys:
                    print("WARNING: Some weights from the checkpoint were not used by the current model structure (ignored):")
                    for key in unexpected_keys:
                        print(f"  - {key}")
            except Exception as e:
                print(f"Error loading raw state_dict for TSM-GRU: {e}")
                return None, {}
            model.eval()
            # Update PHASE_LABELS if needed
            if num_classes != DEFAULT_NUM_CLASSES:
                print(f"Updating PHASE_LABELS for {num_classes} classes.")
                PHASE_LABELS = {i: f"Phase {i}" for i in range(num_classes)}
                if 0 in PHASE_LABELS: PHASE_LABELS[0] = "No Action / Other"
            return model, {}
    except Exception as e:
        print(f"Error loading TSM-GRU model: {e}")
        return None, {}

# --- Frame Processing and Keypoint Extraction ---
def preprocess_frame_for_yolo(frame, clahe):
    """Applies CLAHE enhancement to the frame."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_l = clahe.apply(l)
    enhanced_lab = cv2.merge((clahe_l, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_frame

def extract_hand_keypoints(enhanced_frame, yolo_model):
    """Extracts hand keypoints using YOLO-Pose."""
    results = yolo_model(enhanced_frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
    
    detected_hands_kpts = []  # For drawing all detected hands
    primary_hand_flat_kpts = np.zeros(INPUT_DIM, dtype=np.float32) # For TSM-GRU
    found_primary_hand = False

    if results and len(results) > 0 and results[0].keypoints is not None:
        kpts_data = results[0].keypoints.data # Tensor of shape (num_detected_instances, num_keypoints, 3)
        
        for i in range(kpts_data.shape[0]): # Iterate over detected instances (hands)
            hand_kpts = kpts_data[i].cpu().numpy() # Shape (NUM_HAND_KEYPOINTS, 3)
            
            if hand_kpts.shape[0] == NUM_HAND_KEYPOINTS: # Ensure it's a hand detection
                detected_hands_kpts.append(hand_kpts)
                
                if not found_primary_hand: # Use the first valid hand detection for TSM-GRU
                    # Flatten keypoints: [x1,y1,c1, x2,y2,c2, ...]
                    primary_hand_flat_kpts = hand_kpts.flatten()
                    found_primary_hand = True
            
    return primary_hand_flat_kpts, detected_hands_kpts, found_primary_hand

# --- Drawing Functions ---
def draw_keypoints_on_frame(frame, hands_kpts_list):
    """Draws keypoints for all detected hands."""
    for hand_kpts in hands_kpts_list: # hand_kpts is (NUM_HAND_KEYPOINTS, 3) [x, y, conf]
        # Draw points
        for i in range(hand_kpts.shape[0]):
            x, y, conf = hand_kpts[i]
            if conf > HAND_KP_CONF_THRESHOLD:
                cv2.circle(frame, (int(x), int(y)), 3, COLOR_GREEN, -1)

        # Draw skeleton
        for joint1_idx, joint2_idx in HAND_SKELETON:
            if joint1_idx < hand_kpts.shape[0] and joint2_idx < hand_kpts.shape[0]:
                x1, y1, conf1 = hand_kpts[joint1_idx]
                x2, y2, conf2 = hand_kpts[joint2_idx]

                if conf1 > HAND_KP_CONF_THRESHOLD and conf2 > HAND_KP_CONF_THRESHOLD:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_SKELETON, 2)
    return frame

def display_info(frame, current_phase_name, time_in_phase, quality_assessment, phase_total_durations):
    """Displays phase, time, and quality information on the frame."""
    h, w, _ = frame.shape
    
    # Background for text
    cv2.rectangle(frame, (0, 0), (w, 130), (0,0,0), -1)

    y_offset = 20
    cv2.putText(frame, f"Current Phase: {current_phase_name}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1, cv2.LINE_AA)
    y_offset += 25
    cv2.putText(frame, f"Time in Phase: {time_in_phase:.1f}s", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)
    
    y_offset += 30
    cv2.putText(frame, "Phase Durations (s):", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    
    for phase_name, duration in phase_total_durations.items():
        if phase_name == PHASE_LABELS.get(0, "No Handwashing"): continue # Skip "No Handwashing"
        y_offset += 20
        ideal_time = IDEAL_PHASE_DURATIONS.get(phase_name, 0)
        text = f"- {phase_name}: {duration:.1f}s"
        if ideal_time > 0:
            text += f" (Ideal: {ideal_time}s)"
        cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

    # Display overall quality at the bottom
    cv2.rectangle(frame, (0, h - 30), (w, h), (0,0,0), -1)
    quality_color = COLOR_GREEN if "Good" in quality_assessment or "Excellent" in quality_assessment else COLOR_RED
    cv2.putText(frame, f"Overall Quality: {quality_assessment}", (10, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 1, cv2.LINE_AA)
    
    return frame

# --- Quality Assessment ---
def assess_handwash_quality(phase_total_durations):
    """Assesses the quality of the handwash based on phase durations."""
    if not phase_total_durations or all(d == 0 for d in phase_total_durations.values()):
        return "Not Started or No Phases Recorded"

    issues = []
    completed_phases_count = 0
    good_phases_count = 0

    for phase_name, total_duration in phase_total_durations.items():
        if phase_name == PHASE_LABELS.get(0, "No Handwashing"): # Skip "No Handwashing" phase
            continue

        ideal_duration = IDEAL_PHASE_DURATIONS.get(phase_name, 0)
        
        if total_duration > 0: # Consider phase as attempted
            completed_phases_count +=1
            if ideal_duration > 0:
                if total_duration < ideal_duration * MIN_DURATION_PERCENTAGE:
                    issues.append(f"{phase_name}: Too short ({total_duration:.1f}s vs ideal {ideal_duration}s)")
                else:
                    good_phases_count +=1
            # else: phase has no ideal duration, consider it "done" if > 0
            
    if not issues and completed_phases_count > 0 and completed_phases_count == good_phases_count:
        if completed_phases_count == len(IDEAL_PHASE_DURATIONS):
             return "Excellent! All phases meet ideal durations."
        return "Good! Recorded phases meet ideal durations."
    elif issues:
        return "Needs Improvement: " + "; ".join(issues)
    elif completed_phases_count == 0 and any(name != PHASE_LABELS.get(0) for name in phase_total_durations.keys()):
        return "No significant phases recorded."
    
    return "Assessment Inconclusive."


# --- Main Application ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    yolo_model = load_yolo_model(YOLO_MODEL_PATH, device)
    tsm_gru_model, train_args = load_tsm_gru_model(TSM_GRU_MODEL_PATH, device)

    if yolo_model is None or tsm_gru_model is None:
        print("Failed to load one or more models. Exiting.")
        return

    # Webcam setup
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {WEBCAM_INDEX}.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Keypoint sequence for TSM-GRU
    keypoints_sequence = deque(maxlen=train_args.get('seq_len', SEQUENCE_LENGTH))
    
    # Phase tracking
    current_phase_index = 0 # Default to "No Handwashing"
    # Ensure PHASE_LABELS is correctly populated before this line if it's modified by model loading
    current_phase_name = PHASE_LABELS.get(current_phase_index, "Unknown") 
    current_phase_start_time = time.time()
    time_in_current_phase = 0
    
    # Store total time spent in each phase for quality check
    # This initialization should happen AFTER PHASE_LABELS might have been updated by load_tsm_gru_model
    phase_total_durations = {name: 0.0 for name in PHASE_LABELS.values()}
    
    quality_assessment = "Starting..."
    
    print("Starting webcam feed and handwash monitoring...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        original_frame = frame.copy() # Keep a copy for final display

        # 1. Preprocess frame for YOLO
        enhanced_frame = preprocess_frame_for_yolo(frame, clahe)

        # 2. YOLO Inference: Extract hand keypoints
        primary_hand_kpts_flat, all_detected_hands_kpts, found_hand = extract_hand_keypoints(enhanced_frame, yolo_model)
        
        # Draw all detected hand keypoints
        original_frame = draw_keypoints_on_frame(original_frame, all_detected_hands_kpts)

        # 3. TSM-GRU Inference
        predicted_phase_index = current_phase_index # Default to current if no new prediction

        if found_hand:
            keypoints_sequence.append(primary_hand_kpts_flat)

            if len(keypoints_sequence) == keypoints_sequence.maxlen:
                # Prepare input for TSM-GRU
                sequence_np = np.array(keypoints_sequence) # (SEQUENCE_LENGTH, INPUT_DIM)
                sequence_tensor = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device) # (1, SEQUENCE_LENGTH, INPUT_DIM)
                
                with torch.no_grad():
                    outputs = tsm_gru_model(sequence_tensor) # (1, NUM_CLASSES)
                    _, predicted_tensor = torch.max(outputs.data, 1)
                    predicted_phase_index = predicted_tensor.item()
                # DEBUG: Print predicted phase index
                print(f"Debug TSM-GRU (Hand Found): Raw Predicted Index: {predicted_phase_index}, Current Index: {current_phase_index}, Seq. Full: True")
            else:
                print(f"Debug TSM-GRU (Hand Found): Seq. not full ({len(keypoints_sequence)}/{keypoints_sequence.maxlen}), Current Index: {current_phase_index}")

        else: # No hand found by YOLO
            # Append a frame of zeros to indicate no hand activity
            keypoints_sequence.append(np.zeros(INPUT_DIM, dtype=np.float32))
            
            # If the sequence is full, still try to predict.
            # The model should ideally learn to map sequences of zeros to "No Handwashing".
            if len(keypoints_sequence) == keypoints_sequence.maxlen:
                sequence_np = np.array(keypoints_sequence)
                sequence_tensor = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = tsm_gru_model(sequence_tensor)
                    _, predicted_tensor = torch.max(outputs.data, 1)
                    predicted_phase_index = predicted_tensor.item()
                print(f"Debug TSM-GRU (No Hand): Raw Predicted Index: {predicted_phase_index}, Current Index: {current_phase_index}, Seq. Full: True")
            else:
                 print(f"Debug TSM-GRU (No Hand): Seq. not full ({len(keypoints_sequence)}/{keypoints_sequence.maxlen}), Current Index: {current_phase_index}")
            # The current logic retains the last phase if no new prediction is made (e.g. sequence not full yet)
            # This is acceptable as the sequence filling with zeros should eventually trigger a "No Handwashing" prediction.

        # 4. Update Phase and Timers
        time_in_current_phase = time.time() - current_phase_start_time
        
        if predicted_phase_index != current_phase_index:
            # Phase has changed
            # DEBUG: Print phase change details
            new_phase_name = PHASE_LABELS.get(predicted_phase_index, "Unknown")
            print(f"Debug Phase Change: From '{current_phase_name}' (idx {current_phase_index}) to '{new_phase_name}' (idx {predicted_phase_index})")
            
            phase_total_durations[current_phase_name] += time_in_current_phase # Add duration to previous phase
            
            current_phase_index = predicted_phase_index
            current_phase_name = PHASE_LABELS.get(current_phase_index, "Unknown")
            current_phase_start_time = time.time()
            time_in_current_phase = 0 # Reset timer for new phase
            
            # Re-assess quality when a phase completes
            quality_assessment = assess_handwash_quality(phase_total_durations)

        # 5. Display Information
        display_frame = display_info(original_frame, current_phase_name, time_in_current_phase, quality_assessment, phase_total_durations)
        cv2.imshow("Handwash Monitor", display_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Before quitting, add the duration of the very last phase segment
            phase_total_durations[current_phase_name] += time_in_current_phase
            quality_assessment = assess_handwash_quality(phase_total_durations) # Final assessment
            print("\n--- Final Handwash Report ---")
            for phase, duration in phase_total_durations.items():
                print(f"{phase}: {duration:.2f} seconds")
            print(f"Overall Quality: {quality_assessment}")
            print("-----------------------------")
            break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring stopped.")

if __name__ == "__main__":
    main()