from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model("skin_cancer_model.h5")

# Class names
classes = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions"
]

# Login credentials
USERNAME = "admin"
PASSWORD = "1234"


# Login Page
@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            return redirect(url_for("home"))

    return render_template("login.html")


# Home Page
@app.route("/home")
def home():
    return render_template("home.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    if file.filename == "":
        return redirect(url_for("home"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Image preprocessing
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]

    return render_template(
        "result.html",
        prediction=predicted_class,
        img_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)