from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = load_model("model_cnn.h5")

# =========================
# CONFIG UPLOAD
# =========================
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# pastikan folder ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_names = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

# =========================
# PREDICT FUNCTION
# =========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_index = np.argmax(pred)
    confidence = np.max(pred) * 100

    return class_names[class_index], confidence

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file and file.filename != "":
            filename = file.filename  # 🔥 ambil nama file

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)

            return render_template(
                "index.html",
                prediction=label,
                confidence=round(confidence, 2),
                filename=filename  # 🔥 kirim ke HTML
            )

    return render_template("index.html")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)