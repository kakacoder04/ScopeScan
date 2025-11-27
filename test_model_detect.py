from tensorflow.keras.models import load_model
from ultralytics import YOLO

model = YOLO("./model/model_detect.pt")
model.summary()
