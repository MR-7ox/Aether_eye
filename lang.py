import cv2
import time
import os
import json
import face_recognition
import pyttsx3
from PIL import Image
from dotenv import load_dotenv
from google import genai

# ================= CONFIG =================
MODEL_NAME = "models/gemini-2.5-flash"
MEMORY_FILE = "memory.json"
KNOWN_FACES_DIR = "known_faces"
PLACES_DIR = "places"
LLM_COOLDOWN = 10
# =========================================

# ---------- ENV ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------- TTS ----------
tts = pyttsx3.init()
tts.setProperty("rate", 165)

# ---------- STORAGE ----------
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(PLACES_DIR, exist_ok=True)

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE) as f:
        memory = json.load(f)
else:
    memory = {"people": {}, "places": {}}

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

# ---------- LOAD KNOWN FACES ----------
known_face_encodings = []
known_face_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith((".jpg", ".png")):
        img = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, file))
        enc = face_recognition.face_encodings(img)
        if enc:
            name = file.split("_")[0]
            known_face_encodings.append(enc[0])
            known_face_names.append(name)

print("[INIT] Loaded faces:", known_face_names)

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# ---------- UI BUTTONS ----------
BUTTONS = {
    "person": ((10, 10), (210, 60)),
    "place": ((230, 10), (430, 60)),
    "exit": ((450, 10), (620, 60)),
}

clicked = None

def mouse_event(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, (p1, p2) in BUTTONS.items():
            if p1[0] < x < p2[0] and p1[1] < y < p2[1]:
                clicked = name

cv2.namedWindow("Assistive Agent")
cv2.setMouseCallback("Assistive Agent", mouse_event)

# ---------- STATE ----------
last_llm_time = 0
last_spoken = ""

print("[SYSTEM] Assistant started automatically")

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    detected_person = "Unknown"

    for (top, right, bottom, left), enc in zip(locations, encodings):
        matches = face_recognition.compare_faces(
            known_face_encodings, enc, tolerance=0.6
        )
        if True in matches:
            detected_person = known_face_names[matches.index(True)]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            detected_person,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # ---------- DRAW BUTTONS ----------
    for name, (p1, p2) in BUTTONS.items():
        cv2.rectangle(frame, p1, p2, (50, 50, 50), -1)
        cv2.putText(
            frame,
            name.upper(),
            (p1[0] + 10, p1[1] + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    cv2.imshow("Assistive Agent", frame)

    # ---------- BUTTON ACTIONS ----------
    if clicked == "person":
        name = input("Enter person name: ").strip().lower()
        if name and locations:
            top, right, bottom, left = locations[0]
            face_img = frame[top:bottom, left:right]
            path = f"{KNOWN_FACES_DIR}/{name}_{int(time.time())}.jpg"
            cv2.imwrite(path, face_img)

            enc = face_recognition.face_encodings(rgb, [locations[0]])[0]
            known_face_encodings.append(enc)
            known_face_names.append(name)

            memory["people"][name] = "Known person"
            save_memory()
            print(f"[MEMORY] Person added: {name}")
        else:
            print("[WARN] No face detected")
        clicked = None

    if clicked == "place":
        place = input("Enter place name: ").strip().lower()
        if place:
            path = f"{PLACES_DIR}/{place}_{int(time.time())}.jpg"
            cv2.imwrite(path, frame)
            memory["places"][place] = "Known place"
            save_memory()
            print(f"[MEMORY] Place added: {place}")
        clicked = None

    if clicked == "exit":
        break

    # ---------- GEMINI ----------
    now = time.time()
    if now - last_llm_time > LLM_COOLDOWN:
        img = Image.fromarray(rgb)
        prompt = f"""
You are a visual assistant for a blind user.
Rules:
- One short sentence
- No assumptions
- Describe action and distance only

Detected person: {detected_person}
"""
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, img],
            )
            text = response.text.strip()
            print("[ASSISTANT]", text)

            if text != last_spoken:
                tts.say(text)
                tts.runAndWait()
                last_spoken = text

            last_llm_time = now
        except Exception as e:
            print("[GEMINI ERROR]", e)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------- CLEANUP ----------
cap.release()
cv2.destroyAllWindows()
tts.stop()
print("[SYSTEM] Stopped")