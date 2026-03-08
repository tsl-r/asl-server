from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
from mediapipe.python.solutions import holistic as mp_holistic_module
import base64
import cv2
import json
import threading
import traceback

app = Flask(__name__)
CORS(app)

# ── Threading lock ────────────────────────────
lock = threading.Lock()

# ── Load actions ──────────────────────────────
with open('actions.json', 'r') as f:
    actions = json.load(f)
print(f"Loaded {len(actions)} actions: {actions}")

# ── Model definition ──────────────────────────
class GCN_BiLSTM_Model(nn.Module):
    def __init__(self, input_dim, gcn_hidden_dim, lstm_hidden_dim, output_dim, num_layers=2):
        super(GCN_BiLSTM_Model, self).__init__()
        self.conv1 = GCNConv(input_dim, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.conv3 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.conv4 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm1 = nn.BatchNorm1d(gcn_hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(gcn_hidden_dim)
        self.bilstm = nn.LSTM(
            input_size=gcn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x, edge_index, batch_size):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.conv4(x, edge_index)
        x = x.view(batch_size, -1, x.shape[-1])
        lstm_out, _ = self.bilstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out

# ── Load model ────────────────────────────────
device = torch.device('cpu')
model = torch.load('model.pth', map_location=device, weights_only=False)
model.eval()
print("Model loaded!")

# ── Edges (same as training) ──────────────────
POSE_EDGES = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
              (9,10),(11,12),(11,13),(13,15),(15,17),(12,14),(14,16),(16,18)]
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
              (15,16),(0,17),(17,18),(18,19),(19,20)]
LEFT_HAND_EDGES  = [(u+33, v+33) for u, v in HAND_EDGES]
RIGHT_HAND_EDGES = [(u+54, v+54) for u, v in HAND_EDGES]
CONNECT_EDGES    = [(15,33),(16,54)]
ALL_EDGES = POSE_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES + CONNECT_EDGES
all_edges_bidir = ALL_EDGES + [(v,u) for u,v in ALL_EDGES]
edge_index = torch.tensor(all_edges_bidir, dtype=torch.long).t().contiguous()

# ── MediaPipe setup ───────────────────────────
holistic = mp_holistic_module.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Frame buffer ──────────────────────────────
frame_buffer = []
SEQUENCE_LENGTH = 30

# ── Predict endpoint ──────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    global frame_buffer
    with lock:
        try:
            data = request.json
            if not data or 'image' not in data:
                return jsonify({'error': 'Missing "image" field in request body'}), 400

            image_str = data['image']
            try:
                img_bytes = base64.b64decode(image_str, validate=True)
            except Exception as e:
                return jsonify({'error': f'Invalid base64 image data: {e}'}), 400

            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({'error': 'Failed to decode image, received invalid image data'}), 400

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = holistic.process(frame_rgb)

            # Extract keypoints
            if result.pose_landmarks:
                pose = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark[:33]])
            else:
                pose = np.zeros((33, 3))

            if result.left_hand_landmarks:
                left = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark])
            else:
                left = np.zeros((21, 3))

            if result.right_hand_landmarks:
                right = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark])
            else:
                right = np.zeros((21, 3))

            combined = np.concatenate([pose, left, right], axis=0)  # (75, 3)
            frame_buffer.append(combined)

            # Keep only last 30 frames
            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            # ✅ Pad immediately from the beginning
            padded_buffer = frame_buffer.copy()
            while len(padded_buffer) < SEQUENCE_LENGTH:
                padded_buffer.insert(0, padded_buffer[0])

            # Run model
            x = torch.tensor(np.array(padded_buffer), dtype=torch.float32).view(-1, 3)
            with torch.no_grad():
                output = model(x, edge_index, batch_size=1)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                label = actions[predicted.item()]
                conf = confidence.item()

            # ✅ Reset buffer after prediction
            frame_buffer = []

            if conf < 0.6:
                return jsonify({
                    'translation': '...',
                    'confidence': round(conf * 100, 1),
                    'detected': True
                })

            return jsonify({
                'translation': label,
                'confidence': round(conf * 100, 1),
                'detected': True
            })

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            return jsonify({'error': str(e), 'traceback': tb}), 500

@app.route('/reset', methods=['POST'])
def reset():
    global frame_buffer
    frame_buffer = []
    return jsonify({'status': 'reset'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'actions': actions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)