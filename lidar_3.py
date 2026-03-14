from flask import Flask, request
import numpy as np
import cv2
import os
import time
import ollama
import pyttsx3
import threading
import queue
import serial

# ---------------- CONFIG ----------------

MODEL_NAME = "llava"
PROMPT = "Describe this camera image in two short lines."

SAVE_DIR = "received_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

LIDAR_PORT = "/dev/serial0"
BAUD = 115200

latest_distance = None

# ---------------- LIDAR THREAD ----------------

def lidar_reader():

    global latest_distance

    try:
        ser = serial.Serial(LIDAR_PORT, BAUD, timeout=1)
        print("LiDAR connected")
    except:
        print("LiDAR not found")
        return

    while True:

        data = ser.read(9)

        if len(data) == 9 and data[0] == 0x59 and data[1] == 0x59:
            dist = data[2] + data[3] * 256
            latest_distance = dist

# start lidar thread
threading.Thread(target=lidar_reader, daemon=True).start()


# ---------------- SPEECH QUEUE ----------------

speech_queue = queue.Queue()

def tts_worker():

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    while True:

        text = speech_queue.get()

        if text is None:
            break

        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)

        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):

    speech_queue.put(text)


# ---------------- FLASK ----------------

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():

    global latest_distance

    file = request.files['frame']

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    filename = os.path.join(SAVE_DIR, f"frame_{int(time.time())}.jpg")
    cv2.imwrite(filename, img)

    print("Frame received:", filename)

    # -------- DISTANCE --------

    dist_text = "unknown distance"

    if latest_distance:
        dist_text = f"object distance is {latest_distance} cm"

    print("LiDAR:", dist_text)

    # -------- AI ANALYSIS --------

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": PROMPT + " The detected object is about " + dist_text,
                "images": [filename]
            }
        ]
    )

    text = response["message"]["content"].strip()

    print("AI:", text)

    # -------- SPEAK RESULT --------

    speak(text)

    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)