from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ==============================
# Upload Folder
# ==============================

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==============================
# Load Trained Model
# ==============================

model = tf.keras.models.load_model(
    "model/plant_disease_model.keras"
)


# ==============================
# Load Class Names
# ==============================

with open("model/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


# ==============================
# Disease Information
# ==============================

disease_info = {

    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "description":
            "Late blight is a serious plant disease that can cause dark "
            "spots on leaves and may spread quickly under cool and humid conditions.",
        "prevention":
            "Remove severely affected leaves, avoid overhead watering, "
            "provide good air circulation, and keep the foliage as dry as possible."
    },

    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "description":
            "Early blight commonly causes brown circular spots on tomato "
            "leaves and can lead to yellowing and leaf loss.",
        "prevention":
            "Remove affected leaves, avoid water splashing onto foliage, "
            "maintain proper spacing, and keep the growing area clean."
    },

    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "description":
            "The model detected the uploaded tomato leaf as healthy "
            "with no major disease symptoms.",
        "prevention":
            "Continue proper watering, adequate sunlight, balanced nutrition, "
            "and regularly inspect the plant for changes."
    }
}


# ==============================
# Home Page
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Prediction
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    # Save uploaded image
    filename = secure_filename(file.filename)

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(image_path)

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize((96, 96))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # AI Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction[0])

    predicted_class = class_names[predicted_index]

    confidence = float(
        np.max(prediction[0])
    ) * 100


    # ==============================
    # Get Disease Information
    # ==============================

    info = disease_info.get(
        predicted_class,
        {
            "plant": predicted_class.split("___")[0]
            if "___" in predicted_class
            else "Plant",

            "disease": predicted_class.split("___")[-1]
            if "___" in predicted_class
            else predicted_class,

            "description":
                "The AI model analyzed the uploaded leaf image "
                "and generated the prediction shown above.",

            "prevention":
                "Upload a clear image for better results and "
                "consult an agricultural expert for detailed guidance."
        }
    )


    return render_template(
        "result.html",

        plant=info["plant"],

        disease=info["disease"],

        confidence=round(
            confidence,
            2
        ),

        image_file=filename,

        description=info["description"],

        prevention=info["prevention"]
    )


# ==============================
# Run Application
# ==============================

if __name__ == "__main__":
    app.run(debug=True)