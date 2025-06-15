from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import requests
import json
import os
import csv
from datetime import datetime

app = Flask(__name__, template_folder="templates")

# Roboflow Config
ROBOFLOW_API_KEY = "4dCEXNNecDUWPWHlylMJ"
ROBOFLOW_MODEL = "classify-and-conditionally-detect-single-label-classification-ztjv0"
ROBOFLOW_VERSION = "1"
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"

CSV_FILE = "inference_log.csv"

# Ensure CSV exists with headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "inference", "confidence"])

@app.route('/')
def show_result():
    entries = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline='') as f:
            reader = csv.DictReader(f)
            entries = list(reader)[-20:]  # Show only the last 20 entries
    return render_template("result.html", entries=entries)

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.data:
        try:
            img = Image.open(io.BytesIO(request.data)).convert("RGB")
            img = img.resize((640, 640))

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)

            print("📤 Sending to Roboflow...")

            response = requests.post(
                ROBOFLOW_URL,
                files={"file": buffer},
                data={"name": "esp32_upload"}
            )

            print("📩 Roboflow responded:", response.text)

            result = response.json()

            if 'predictions' in result and len(result['predictions']) > 0:
                top_prediction = result['predictions'][0]
                inference = top_prediction.get('class', 'Unknown')
                confidence = str(round(top_prediction.get('confidence', 0) * 100, 2))
            else:
                inference = "Couldn't identify"
                confidence = "0.0"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, inference, confidence])

            return response.text, 200, {'Content-Type': 'text/plain'}

        except Exception as e:
            print("🔥 Error processing image:", e)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No image data received"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
