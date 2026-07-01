from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# --- Trik bypass error quantization_config Keras ---
from keras.src.layers.core.dense import Dense
original_dense_init = Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None) # Buang parameter penyebab error jika ada
    original_dense_init(self, *args, **kwargs)
Dense.__init__ = patched_dense_init
# --------------------------------------------------

#load model
MODEL_PATH = "model/model_klasifikasi_buah_final_v2.keras"

model = load_model(MODEL_PATH)

print("\n=== MobileNetV2 Loaded ===")
print("Input :", model.input_shape)
print("Output:", model.output_shape)


# load model
with open("labels.txt", "r") as f:

    class_names = [

        line.strip()

        for line in f.readlines()

    ]

print("Class :", class_names)

# load si yolov8
yolo = YOLO("yolov8n.pt")

# menentukan nama buah

FRUIT_IDS = {

    46: "banana",

    47: "apple",

    49: "orange"

}

print("\n=== YOLO Loaded ===")

# preprocessing mobilenetv2

IMG_SIZE = (150,150)

def preprocess_image(image):
    image = image.convert("RGB")
    
    # padding
    target_size = (150, 150)
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, (255, 255, 255))
    new_img.paste(
        image,
        (
            (target_size[0] - image.size[0]) // 2,
            (target_size[1] - image.size[1]) // 2
        )
    )
    
    img_array = np.array(new_img)
    img_array = np.expand_dims(img_array, axis=0)
    # ini tidak normalisasi alias disamain sama yang training dicolab
    return img_array

# deteksi use yolo dan juga klasifikasi pakai mobilenet
def detect_and_classify(image_path):
    image = cv2.imread(image_path)
    results = yolo(image, verbose=False)

    # 1. Kumpulkan semua box per jenis buah, simpan beserta yolo confidence-nya
    candidates = {}  # fruit_name -> box dengan conf yolo tertinggi
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in FRUIT_IDS:
                continue

            fruit_name = FRUIT_IDS[cls_id]
            yolo_conf = float(box.conf[0])

            if fruit_name not in candidates or yolo_conf > candidates[fruit_name]["yolo_conf"]:
                candidates[fruit_name] = {
                    "box": box,
                    "yolo_conf": yolo_conf
                }

    # 2. Proses hanya 1 box terbaik per jenis buah
    predictions = []
    detected_summary = {}
    detected_count = 0
    annotated_image = image.copy()

    for fruit_name, data in candidates.items():
        box = data["box"]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        img = preprocess_image(crop_pil)

        pred = model.predict(img, verbose=0)
        class_index = np.argmax(pred)
        confidence = float(np.max(pred))
        ripeness = class_names[class_index]

        detected_count += 1
        detected_summary[fruit_name] = 1  # selalu 1 karena cuma 1 wakil

        predictions.append({
            "fruit": fruit_name,
            "ripeness": ripeness,
            "confidence": confidence
        })

        if ripeness == "ripe":
            color = (0, 255, 0)
        elif ripeness == "unripe":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        label = f"{fruit_name} | {ripeness} | {confidence*100:.1f}%"
        cv2.putText(
            annotated_image, label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )

    detected_path = None
    if detected_count > 0:
        detected_path = os.path.join(app.config["UPLOAD_FOLDER"], "detected.jpg")
        cv2.imwrite(detected_path, annotated_image)

    return predictions, detected_summary, detected_count, detected_path


@app.route("/", methods=["GET", "POST"])
def index():

    image_file = None
    detected_image = None
    predictions = []
    detected_summary = {}
    detected_count = 0
    if request.method == "POST":
        if "image" not in request.files:
            return render_template(
                "index.html",
                image_file=None,
                detected_image=None,
                predictions=[],
                detected_summary={},
                detected_count=0
            )

        file = request.files["image"]
        if file.filename == "":
            return render_template(
                "index.html",
                image_file=None,
                detected_image=None,
                predictions=[],
                detected_summary={},
                detected_count=0
            )

        filename = file.filename
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)
        image_file = filepath
        (
            predictions,
            detected_summary,
            detected_count,
            detected_image
        ) = detect_and_classify(filepath)

    return render_template(
        "index.html",
        image_file=image_file,
        detected_image=detected_image,
        predictions=predictions,
        detected_summary=detected_summary,
        detected_count=detected_count
    )

if __name__ == "__main__":
    app.run(
        debug=True
    )
