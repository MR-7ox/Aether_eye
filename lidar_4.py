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
        print("[LIDAR] Serial connected")
    except Exception as e:
        print("[LIDAR ERROR]", e)
        return

    while True:

        try:

            # read first header byte
            byte = ser.read(1)

            if byte == b'\x59':

                # read second header byte
                byte2 = ser.read(1)

                if byte2 == b'\x59':

                    frame = ser.read(7)

                    if len(frame) == 7:

                        dist_l = frame[0]
                        dist_h = frame[1]

                        distance = dist_l + dist_h * 256

                        latest_distance = distance

                        print("[LIDAR]", distance, "cm")

        except Exception as e:

            print("[LIDAR ERROR]", e)


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
            print("[TTS ERROR]", e)

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

    print("[FRAME]", filename)

    # ---------- DISTANCE ----------

    if latest_distance is None:
        dist_text = "distance unavailable"
    else:
        dist_text = f"{latest_distance} cm"

    print("[LIDAR TEXT]", dist_text)

    # ---------- AI ----------

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": PROMPT + " The object is about " + dist_text,
                "images": [filename]
            }
        ]
    )

    text = response["message"]["content"].strip()

    print("[AI]", text)

    speak(text)

    return "OK"


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":

    print("[SERVER] Flask server started")

    app.run(host="0.0.0.0", port=5000, threaded=True)