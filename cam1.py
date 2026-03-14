import cv2
import time
import os
import numpy as np
import face_recognition
from PIL import Image
from dotenv import load_dotenv
from google import genai
import pyttsx3

# ------------------ LOAD ENV ------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# ------------------ GEMINI CLIENT ------------------
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "models/gemini-2.5-flash"

# ------------------ TEXT TO SPEECH ------------------
tts = pyttsx3.init()
tts.setProperty("rate", 170)   # speaking speed (friendly)
tts.setProperty("volume", 1.0)

# ------------------ LOAD KNOWN FACES ------------------
known_face_encodings = []
known_face_names = []

if not os.path.exists("known_faces"):
    raise RuntimeError("❌ known_faces folder not found")

for file in os.listdir("known_faces"):
    path = os.path.join("known_faces", file)
    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_face_encodings.append(encodings[0])
        name = file.split("_")[0]   # deva_1.jpg → deva
        known_face_names.append(name)

print("✅ Loaded known faces:", known_face_names)

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible")

print("✅ Visual Assistance System Started (Press Q to quit)")

last_spoken = ""   # prevents repeating same sentence

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------ FACE RECOGNITION ------------------
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected_person = "Unknown"

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding,
            tolerance=0.65
        )
        if True in matches:
            detected_person = known_face_names[matches.index(True)]
            break

    # ------------------ GEMINI ------------------
    img = Image.fromarray(rgb_frame)

    context = f"""
You are a visual assistant for a blind user.

Detected person: {detected_person}

Speak in ONE short, friendly sentence.
Be conversational and reassuring.
Mention the person by name if known.
Focus only on what is important right now.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[context, img]
    )

    text = response.text.strip()
    print("🧠 Assistant:", text)

    # ------------------ SPEAK (ONLY IF NEW) ------------------
    if text and text != last_spoken:
        tts.say(text)
        tts.runAndWait()
        last_spoken = text

    # Quit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(3)  # slower loop for comfort

# ------------------ CLEANUP ------------------
cap.release()
cv2.destroyAllWindows()
tts.stop()
print("🛑 Visual Assistance System Stopped")