from ultralytics import YOLO
import os

WEIGHTS = "runs/detect/runs/finetune_yolo26/weights/best.pt"
VIDEO = "source/stug.mp4"
CONF = 0.1
CLASSES = [32]

print("Using weights:", WEIGHTS)
model = YOLO(WEIGHTS)

try:
    model.predict(source=VIDEO, save=True, device="0", conf=CONF, classes=CLASSES)
    print("Inference completed on GPU.")
except Exception as e:
    print("GPU inference failed, retrying on CPU. Error:", e)
    model.predict(source=VIDEO, save=True, device="cpu", conf=CONF, classes=CLASSES)
    print("Inference completed on CPU.")

print("Check runs/detect/predict/ for outputs.")
