import threading
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Initialize YOLO model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov5su.pt').to(device)

def train_model(yaml_file_path, epochs, batch_size):
    try:
        model.train(
            data=yaml_file_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=320,
            device=device,
            workers=8,
            half=True,
            save_period=1
        )
    except Exception as e:
        print(f"Training failed: {e}")

@app.route('/')
def home():
    return "Welcome to the YOLOv5 Training and Inference API!"


@app.route('/train', methods=['POST'])
def start_training():
    try:
        data = request.json
        yaml_file_path = data['yaml_file_path']
        epochs = data.get('epochs', 1)
        batch_size = data.get('batch_size', 4)
        training_thread = threading.Thread(target=train_model, args=(yaml_file_path, epochs, batch_size))
        training_thread.start()
        return jsonify({'message': 'Training started'}), 200
    except KeyError as e:
        return jsonify({'error': f'Missing parameter: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_app():
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    run_app()
