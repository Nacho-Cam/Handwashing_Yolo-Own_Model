import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os
from collections import deque
from flask import Flask, Response, render_template_string
from phases import PHASES

# --- Configuration ---
# Adjust these paths if main_app.py is not in the project root or if models are elsewhere
DEFAULT_TSM_MODEL_PATH = 'lavadodemanosyoloposev11yentrenamiento/best_tsm_gru.pth'
DEFAULT_YOLO_MODEL_PATH = 'lavadodemanosyoloposev11yentrenamiento/yolo11n-pose-hands.pt' # Fine-tuned model

# Import TSM_GRU model from the subdirectory
try:
    from lavadodemanosyoloposev11yentrenamiento.model.tsm_gru import TSM_GRU
except ImportError as e:
    print(f"Error importing TSM_GRU: {e}")
    print("Make sure lavadodemanosyoloposev11yentrenamiento/model/tsm_gru.py exists and is correct.")
    TSM_GRU = None

# Feature dimension should match the TSM-GRU model's training
EXPECTED_FEATURE_DIM = 63 # For 21 hand keypoints (yolo11n-pose-hands.pt)

# These might be updated from the model checkpoint
SEQUENCE_LENGTH = 150
TARGET_FPS = 10 # Reduced for web streaming example

LABEL_NAMES = {i: name for i, name in enumerate(PHASES)}
LABEL_NAMES[-1] = 'Iniciando...'
LABEL_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 128, 0),
    -1: (128, 128, 128)
}

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global Variables for Models (Load once) ---
tsm_gru_model_global = None
yolo_model_global = None
device_global = None
clahe_global = None
model_input_dim_global = EXPECTED_FEATURE_DIM
sequence_length_global = SEQUENCE_LENGTH # Will be updated from checkpoint if possible

def load_models():
    global tsm_gru_model_global, yolo_model_global, device_global, clahe_global
    global model_input_dim_global, sequence_length_global, LABEL_NAMES

    if tsm_gru_model_global is not None and yolo_model_global is not None:
        return True # Models already loaded

    if TSM_GRU is None:
        print("TSM_GRU class not loaded. Cannot initialize model.")
        return False

    device_global = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_global}")

    # Load TSM-GRU Model
    tsm_model_path_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_TSM_MODEL_PATH)
    if not os.path.exists(tsm_model_path_abs):
        print(f"Error: TSM-GRU model not found at {tsm_model_path_abs}")
        return False
    try:
        checkpoint = torch.load(tsm_model_path_abs, map_location=device_global)
        # --- PATCH: Handle both checkpoint dict and raw state_dict ---
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_args = checkpoint.get('args', {})
            model_input_dim_global = model_args.get('input_dim', EXPECTED_FEATURE_DIM)
            num_classes = model_args.get('num_classes', 7)
            train_seq_len = model_args.get('seq_len', sequence_length_global)
            if train_seq_len != sequence_length_global:
                print(f"Updating SEQUENCE_LENGTH from {sequence_length_global} to {train_seq_len} based on model checkpoint.")
                sequence_length_global = train_seq_len
            tsm_gru_model_global = TSM_GRU(
                input_dim=model_input_dim_global,
                hidden_dim=model_args.get('hidden_dim', 256),
                num_classes=num_classes,
                num_layers_gru=model_args.get('gru_layers', 2),
                tsm_segments=model_args.get('tsm_segments', 8),
                tsm_shift_div=model_args.get('tsm_shift_div', 3),
                bidirectional=model_args.get('bidirectional', False),
                use_attention=model_args.get('use_attention', False),
                dropout=model_args.get('dropout', 0.0)
            ).to(device_global)
            tsm_gru_model_global.load_state_dict(checkpoint['model_state_dict'], strict=False)
            tsm_gru_model_global.eval()
            print(f"TSM-GRU model loaded. Expecting input_dim: {model_input_dim_global}, seq_len: {sequence_length_global}")
        else:
            # Assume raw state_dict (no args, use defaults)
            print("INFO: Loaded raw state_dict (no checkpoint dictionary). Attempting to infer num_classes from state_dict.")
            num_classes = 7
            if 'fc.weight' in checkpoint:
                num_classes = checkpoint['fc.weight'].shape[0]
            elif 'fc.bias' in checkpoint:
                num_classes = checkpoint['fc.bias'].shape[0]
            print(f"INFO: Inferred num_classes = {num_classes}")
            tsm_gru_model_global = TSM_GRU(
                input_dim=EXPECTED_FEATURE_DIM,
                hidden_dim=256,
                num_classes=num_classes,
                num_layers_gru=2,
                tsm_segments=8,
                tsm_shift_div=3,
                bidirectional=False,
                use_attention=False,
                dropout=0.0
            ).to(device_global)
            tsm_gru_model_global.load_state_dict(checkpoint, strict=False)
            tsm_gru_model_global.eval()
            print(f"TSM-GRU model loaded from raw state_dict. Expecting input_dim: {EXPECTED_FEATURE_DIM}, seq_len: {sequence_length_global}")
        # Update LABEL_NAMES if num_classes changed
        LABEL_NAMES = {i: f"Phase {i}" for i in range(num_classes)}
        LABEL_NAMES[-1] = 'Iniciando...'
    except Exception as e:
        print(f"Error loading TSM-GRU model: {e}")
        tsm_gru_model_global = None
        return False

    # Load YOLO Model
    yolo_model_path_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_YOLO_MODEL_PATH)
    if not os.path.exists(yolo_model_path_abs):
        print(f"Error: YOLO model not found at {yolo_model_path_abs}")
        return False
    try:
        yolo_model_global = YOLO(yolo_model_path_abs)
        yolo_model_global.to(device_global)
        print("YOLO model loaded.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        yolo_model_global = None
        return False

    clahe_global = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return True

def extract_keypoints_frame_web(frame, yolo_model, clahe, expected_dim):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_frame_gray = clahe.apply(gray)
    clahe_frame_bgr = cv2.cvtColor(clahe_frame_gray, cv2.COLOR_GRAY2BGR)
    results = yolo_model(clahe_frame_bgr, verbose=False)
    
    frame_kpts = np.zeros(expected_dim)

    if results and results[0].keypoints is not None:
        kpts_data = results[0].keypoints.data.cpu().numpy()
        if kpts_data.shape[0] > 0:
            actual_kpts_flat = kpts_data[0].flatten()
            if len(actual_kpts_flat) >= expected_dim:
                frame_kpts = actual_kpts_flat[:expected_dim]
            else:
                frame_kpts[:len(actual_kpts_flat)] = actual_kpts_flat
    return frame_kpts, results[0].plot() if results and results[0] else frame

def generate_frames():
    if not load_models(): # Try to load models if not already loaded
        # Models failed to load, stream an error message
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Error: Models not loaded.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1) # Avoid busy loop

    # VIDEO SOURCE:
    # For Render, you'll need to provide a video file.
    # Upload a video (e.g., "sample_video.mp4") with your project.
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_video.mp4")
    
    cap = None
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video file: {video_path}")
    else:
        print(f"Video file not found: {video_path}. Using webcam.")
        cap = cv2.VideoCapture(0)  # Use default webcam

    dummy_frame_counter = 0
    max_dummy_frames = 300 

    keypoints_buffer = deque(maxlen=sequence_length_global)
    last_phase_prediction = -1

    while True:
        frame = None
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video file. Resetting or stopping.")
                # Option 1: Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                ret, frame = cap.read()
                if not ret: # If reset also fails, stop
                    break 
                # Option 2: Stop after one pass (comment above, uncomment break below)
                # break 
        
        if frame is None: # Fallback to dummy frames if cap failed or no video
            frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Dummy Frame {dummy_frame_counter}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            dummy_frame_counter = (dummy_frame_counter + 1) % max_dummy_frames
            if dummy_frame_counter == 0: print("Looping dummy frames...")


        current_keypoints, annotated_frame = extract_keypoints_frame_web(
            frame, yolo_model_global, clahe_global, model_input_dim_global
        )
        keypoints_buffer.append(current_keypoints)

        predicted_phase_index = last_phase_prediction
        if len(keypoints_buffer) == sequence_length_global:
            sequence_np = np.array(keypoints_buffer)
            sequence_tensor = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device_global)
            with torch.no_grad():
                outputs = tsm_gru_model_global(sequence_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_phase_index = predicted.item()
            last_phase_prediction = predicted_phase_index
        
        phase_name = LABEL_NAMES.get(last_phase_prediction, "Desconocido")
        color = LABEL_COLORS.get(last_phase_prediction, (255, 255, 255))
        cv2.putText(annotated_frame, f"Fase: {phase_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue # Skip frame if encoding failed
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1.0 / TARGET_FPS)
    
    if cap:
        cap.release()

@app.route('/')
def index():
    return render_template_string('''
    <html>
        <head><title>Handwash Monitor</title></head>
        <body>
            <h1>Handwash Monitor Stream</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body>
    </html>''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Gunicorn will run the app, so this is mainly for local testing.
# The load_models() call is deferred to the first request to generate_frames
# to avoid issues with Gunicorn's multiple worker startup.
if __name__ == '__main__':
    print("Starting Flask app for local testing...")
    # For local testing, models can be loaded at startup.
    # However, for Gunicorn, it's better to load them lazily or in a post_fork hook.
    # load_models() # Deferred to first request in generate_frames
    app.run(host='0.0.0.0', port=5001, debug=False) # Use a different port like 5001 for local
