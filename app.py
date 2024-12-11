import os
import json
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image, ImageDraw
import numpy as np
from inference import get_model

# Initialize Flask app
app = Flask(__name__)

# Replace with your model ID from Roboflow
MODEL_ID = "retail-shelf-product-detection-x9us2"
model = get_model(model_id=MODEL_ID)

# Route to display the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle JSON predictions
@app.route('/predict_json', methods=['POST'])
def predict_json():
    if 'file' not in request.files:
        return jsonify({"error": "No file part provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image
        img = Image.open(file.stream).convert("RGB")
        img = np.array(img)

        # Run inference
        results = model.predict(img, confidence=1, overlap=30).json()

        # Process detections
        processed_detections = []
        for det in results["predictions"]:
            # Convert bounding box to [x_min, y_min, x_max, y_max]
            bbox = [
                det["x"] - det["width"] / 2,
                det["y"] - det["height"] / 2,
                det["x"] + det["width"] / 2,
                det["y"] + det["height"] / 2,
            ]
            processed_detections.append({
                "class": det["class"],
                "confidence": det["confidence"],
                "bbox": bbox
            })

        return jsonify({
            "message": "Inference completed",
            "detections": processed_detections
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to visualize image with bounding boxes
@app.route('/visualize', methods=['POST'])
def visualize():
    if 'file' not in request.files:
        return jsonify({"error": "No file part provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image
        img = Image.open(file.stream).convert("RGB")
        img_array = np.array(img)

        # Run inference
        results = model.predict(img_array, confidence=1, overlap=30).json()

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        for det in results["predictions"]:
            # Get bounding box
            bbox = [
                det["x"] - det["width"] / 2,
                det["y"] - det["height"] / 2,
                det["x"] + det["width"] / 2,
                det["y"] + det["height"] / 2,
            ]
            draw.rectangle(bbox, outline="red", width=3)
            draw.text((bbox[0], bbox[1]), det["class"], fill="red")

        # Save the image to a temporary file
        output_path = "output.jpg"
        img.save(output_path)

        # Return the image to the user
        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
