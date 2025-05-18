import os
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://raifazhar.github.io"])  # ⬅️ Allow only your frontend
model = None  # Lazy loading


def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model("game_cover_model.keras")
        print("Model loaded.")


def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["GET"])
def home():
    return "✅ GameCoverAI Backend is running"


@app.route("/predict", methods=["POST"])
def predict():
    load_model()

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    try:
        img = Image.open(file.stream)
        img = img.convert("RGB")  # force RGB
        processed_img = preprocess_image(img)
        print("Processed image shape:", processed_img.shape)
    except Exception as e:
        print("Image processing failed:", e)
        return jsonify({"error": "Invalid image"}), 400

    try:
        prediction = model.predict(processed_img)[0]
        print("Prediction result:", prediction)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        print("Prediction failed:", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
