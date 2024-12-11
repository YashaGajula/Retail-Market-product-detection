import os
import json
from roboflow import Roboflow
from PIL import Image
import numpy as np

# Initialize the Roboflow model
rf = Roboflow(api_key="D19FVfXVsgB2ZEVrOP6F")  # Replace with your API key
workspace = "team3-ohd3p"  # Replace with your workspace name
model_id = "retail-shelf-product-detection-x9us2"
model = rf.workspace(workspace).project(model_id).version(4).model

# Folder path containing the images
folder_path = r"C:\Users\YASHASWINI\OneDrive\Documents\project_folder_2\sample_images"  # Replace with your folder path

# List to store results for all images
json_results = []

# Loop through all the images in the folder
for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)

    # Process only image files (jpg, jpeg, png)
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            print(f"Processing {img_name}")  # Debugging log
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)

            # Run inference
            results = model.predict(img, confidence=10, overlap=30).json()

            # Print raw predictions for debugging
            print(f"Raw Predictions for {img_name}: {results['predictions']}")

            # Process detections and remove the image_path (which is a numpy array)
            processed_detections = []

            for det in results["predictions"]:
                # Remove the ndarray (image_path) from the prediction
                if "image_path" in det:
                    del det["image_path"]

                # Convert bounding box to [x_min, y_min, x_max, y_max]
                bbox = [
                    det["x"] - det["width"] / 2,
                    det["y"] - det["height"] / 2,
                    det["x"] + det["width"] / 2,
                    det["y"] + det["height"] / 2,
                ]
                det["bbox"] = bbox

                processed_detections.append(det)

            # Append results to JSON response for the current image
            json_results.append({
                "image_name": img_name,
                "detections": processed_detections
            })

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            json_results.append({"image_name": img_name, "error": str(e)})

# Save results to a JSON file
output_file = "predictions_output_batch.json"
with open(output_file, "w") as json_file:
    json.dump(json_results, json_file, indent=4)

print(f"Batch predictions saved to {output_file}")
