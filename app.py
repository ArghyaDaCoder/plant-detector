from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import io
import requests
import json
import os
import csv
from datetime import datetime
import uuid

app = Flask(__name__, template_folder="templates", static_folder="static")

# Roboflow Config
ROBOFLOW_API_KEY = "4dCEXNNecDUWPWHlylMJ"
ROBOFLOW_MODEL = "classify-and-conditionally-detect-single-label-classification-ztjv0"
ROBOFLOW_VERSION = "1"
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"

CSV_FILE = "inference_log.csv"
IMAGE_FOLDER = "static/images"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Ensure CSV exists with headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "inference", "confidence", "image_name"])

def time_ago(timestamp_str):
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        now = datetime.utcnow()
        diff = now - timestamp
        minutes = int(diff.total_seconds() / 60)
        if minutes < 1:
            return "Just now"
        elif minutes == 1:
            return "1 minute ago"
        elif minutes < 60:
            return f"{minutes} minutes ago"
        elif minutes < 120:
            return "1 hour ago"
        else:
            hours = minutes // 60
            return f"{hours} hours ago"
    except:
        return "Unknown time"

@app.route('/')
def show_result():
    return render_template("result.html")

@app.route('/data')
def get_data():
    entries = []
    details_map = {}

    # Load plant disease info
    with open("plant_info.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            codename = row["Codename"]
            details_map[codename] = row

    # Load inference logs
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline='') as f:
            reader = csv.DictReader(f)
            for row in list(reader)[-20:]:
                codename = row["inference"]
                info = details_map.get(codename, {
                    "Plant_Name": "Unknown",
                    "Disease": "Unknown",
                    "Description": "No info available.",
                    "Cure": "N/A"
                })

                entry = {
                    "timestamp": time_ago(row["timestamp"]),
                    "image_url": f"/static/images/{row['image_name']}",
                    "Plant_Name": info["Plant_Name"],
                    "Disease": info["Disease"],
                    "Description": info["Description"],
                    "Cure": info["Cure"]
                }

                entries.append(entry)
    return jsonify(entries)


@app.route('/upload', methods=['POST'])
def upload_image():
    if request.data:
        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            original_name = f"{uuid.uuid4().hex}.jpg"
            original_path = os.path.join(IMAGE_FOLDER, original_name)

            # Save and prepare image
            img = Image.open(io.BytesIO(request.data)).convert("RGB")
            img.save(original_path, format="JPEG", quality=60)

            # 👉 1. Send to Leaf Detection AI (model #1)
            leaf_model_url = "https://detect.roboflow.com/leaf-detection-x2pwn/1?api_key=4dCEXNNecDUWPWHlylMJ"
            with open(original_path, "rb") as f:
                leaf_response = requests.post(leaf_model_url, files={"file": f})
            boxes = leaf_response.json().get("predictions", [])

            if not boxes:
                print("🚫 No leaves detected")
                return jsonify({"error": "No leaves found"}), 200

            image_cv = cv2.imread(original_path)
            all_logs = []

            # Folder for cropped images
            cropped_folder = os.path.join(IMAGE_FOLDER, "temp")
            os.makedirs(cropped_folder, exist_ok=True)

            for i, pred in enumerate(boxes):
                # 📦 Crop leaf
                x = int(pred["x"] - pred["width"] / 2)
                y = int(pred["y"] - pred["height"] / 2)
                w = int(pred["width"])
                h = int(pred["height"])
                crop = image_cv[y:y+h, x:x+w]

                crop_name = f"crop_{uuid.uuid4().hex}.jpg"
                crop_path = os.path.join(cropped_folder, crop_name)
                cv2.imwrite(crop_path, crop)

                # 👉 2. Send cropped leaf to Classification AI (model #2)
                with open(crop_path, "rb") as cf:
                    classify_response = requests.post(
                        ROBOFLOW_URL,
                        files={"file": cf},
                        data={"name": f"leaf_crop_{i}"}
                    )

                classify_result = classify_response.json()
                if 'predictions' in classify_result and len(classify_result['predictions']) > 0:
                    top_pred = classify_result['predictions'][0]
                    inference = top_pred.get('class', 'Unknown')
                    confidence = str(round(top_pred.get('confidence', 0) * 100, 2))
                else:
                    inference = "Couldn't identify"
                    confidence = "0.0"

                all_logs.append([timestamp, inference, confidence, original_name])

                # 🧹 Delete cropped image
                os.remove(crop_path)

            # Delete temp folder
            os.rmdir(cropped_folder)

            # 📝 Log all entries
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(all_logs)

            return jsonify({"status": "done", "detected_leaves": len(all_logs)}), 200

        except Exception as e:
            print("🔥 Error:", e)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No image data received"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
