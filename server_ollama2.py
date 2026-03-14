from flask import Flask, request
import numpy as np
import cv2
import os
import time
import ollama
import pyttsx3
import threading
import queue
from ultralytics import YOLO

# ---------------- CONFIG ----------------

MODEL_NAME = "llava"

PROMPT = """
Describe the important activity in this CCTV camera frame.
Focus on people, animals, or moving objects.
Reply in ONE short sentence.
"""

SAVE_DIR = "received_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# YOLO object detector
yolo = YOLO("yolov8n.pt")

important_objects = [
    "person","dog","cat","car","bicycle","motorcycle","truck"
]

# ---------------- TTS ----------------

speech_queue = queue.Queue()

def tts_worker():

    engine = pyttsx3.init()
    engine.setProperty("rate",180)

    while True:

        text = speech_queue.get()

        if text is None:
            break

        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:",e)

        speech_queue.task_done()

threading.Thread(target=tts_worker,daemon=True).start()

def speak(text):
    speech_queue.put(text)

# ---------------- FRAME LOGIC ----------------

frame_buffer = []
prev_frame = None

change_threshold = 180000
speech_cooldown = 5
last_speech_time = 0

# ---------------- FLASK ----------------

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():

    global frame_buffer
    global prev_frame
    global last_speech_time

    file = request.files["frame"]

    data = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    filename = os.path.join(SAVE_DIR,f"frame_{int(time.time())}.jpg")
    cv2.imwrite(filename,frame)

    print("Frame received:",filename)

    # ---- collect 3 frames ----

    frame_buffer.append(frame)

    if len(frame_buffer) < 3:
        return "OK"

    frame = frame_buffer[1]
    frame_buffer = []

    # ---- motion comparison ----

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if prev_frame is None:
        prev_frame = gray
        return "OK"

    diff = cv2.absdiff(prev_frame,gray)
    change_score = diff.sum()

    prev_frame = gray

    if change_score < change_threshold:
        print("Small change, skipping AI")
        return "OK"

    print("Scene changed, detecting objects...")

    # ---------------- YOLO OBJECT DETECTION ----------------

    results = yolo(frame)

    detected_objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = yolo.names[cls]
            detected_objects.append(label)

    print("Objects:", detected_objects)

    if not any(obj in important_objects for obj in detected_objects):
        print("No important objects")
        return "OK"

    # ---------------- AI ANALYSIS ----------------

    temp_file = os.path.join(SAVE_DIR,"analysis.jpg")
    cv2.imwrite(temp_file,frame)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{
            "role":"user",
            "content":PROMPT,
            "images":[temp_file]
        }]
    )

    text = response["message"]["content"].strip()

    print("AI:",text)

    # ---------------- SPEECH ----------------

    now = time.time()

    if now - last_speech_time > speech_cooldown:
        speak(text)
        last_speech_time = now

    return "OK"


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,threaded=True)