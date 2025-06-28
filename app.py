from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import io
import requests
import json
import os
import csv
from datetime import datetime
import uuid
import cv2
from threading import Thread

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
            image_name = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(IMAGE_FOLDER, image_name)

            img = Image.open(io.BytesIO(request.data)).convert("RGB")
            img = img.resize((640, 640))  # better match for YOLO
            img.save(image_path, format="JPEG", quality=85)

            # Run background detection/classification
            Thread(target=process_in_background, args=(image_path, timestamp, image_name)).start()

            return jsonify({"status": "Image uploaded successfully. Processing will continue in background."})

        except Exception as e:
            print("🔥 Error in upload_image:", e)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No image data received"}), 400


def process_in_background(image_path, timestamp, image_name):
    try:
        print("🧠 Background task started")

        print("🖼️ Image saved at:", image_path)
        image_cv = cv2.imread(image_path)
        print("🧪 Image shape:", image_cv.shape)

        # 🔍 Send to object detector
        with open(image_path, "rb") as original:
            detect_response = requests.post(
                "https://detect.roboflow.com/leaf-detection-x2pwn/1?api_key=4dCEXNNecDUWPWHlylMJ",
                files={"file": original}
            )
        detect_result = detect_response.json()

        print("🔍 Full Detection Response:\n", json.dumps(detect_result, indent=2))

        # Filter predictions by confidence
        predictions = [p for p in detect_result.get("predictions", []) if p["confidence"] > 0.5]

        print(f"📦 Detected {len(predictions)} leaves with >50% confidence")

        for i, pred in enumerate(predictions):
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            crop = image_cv[y - h//2 : y + h//2, x - w//2 : x + w//2]
            crop_path = os.path.join("static/images", f"{uuid.uuid4().hex}_crop.jpg")
            cv2.imwrite(crop_path, crop)
            print(f"🌱 Cropped leaf saved at: {crop_path}")

            with open(crop_path, "rb") as cf:
                classify_response = requests.post(
                    f"https://detect.roboflow.com/YOUR_CLASSIFIER_MODEL_HERE/1?api_key=4dCEXNNecDUWPWHlylMJ",  # ⛳ TODO
                    files={"file": cf}
                )
                classify_result = classify_response.json()
                print("🔍 Classification Result:", json.dumps(classify_result, indent=2))

            label = classify_result['predictions'][0]['class']
            confidence = classify_result['predictions'][0]['confidence']

            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, label, round(confidence*100, 2), image_name])
        
        print("✅ Background task done")
    
    except Exception as e:
        print("💥 Error in background task:", e)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
