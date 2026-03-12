from flask import Flask, request
import numpy as np
import cv2
import os
import time

app = Flask(__name__)

SAVE_DIR = "received_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['frame']
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    filename = os.path.join(SAVE_DIR, f"frame_{int(time.time())}.jpg")
    cv2.imwrite(filename, img)

    print("Frame received:", filename)

    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)