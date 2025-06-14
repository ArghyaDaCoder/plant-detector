from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import requests
import json
import os

app = Flask(__name__, template_folder="templates")

# Roboflow Config
ROBOFLOW_API_KEY = "4dCEXNNecDUWPWHlylMJ"
ROBOFLOW_MODEL = "classify-and-conditionally-detect-single-label-classification-ztjv0"
ROBOFLOW_VERSION = "1"
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"

@app.route('/')
def show_result():
    if os.path.exists("result.json"):
        with open("result.json", "r") as f:
            result = json.load(f)
    else:
        result = {
            "top": "Waiting for ESP32-CAM upload...",
            "confidence": "0.0",
            "predictions": []
        }
    return render_template("result.html", result=result)

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
            with open("result.json", "w") as f:
                json.dump(result, f)

            return response.text, 200, {'Content-Type': 'text/plain'}

        except Exception as e:
            print("🔥 Error processing image:", e)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No image data received"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
