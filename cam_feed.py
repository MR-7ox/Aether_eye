import cv2
import time
import os
from PIL import Image
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# ✅ Create client (NEW way)
client = genai.Client(api_key=GEMINI_API_KEY)

# ✅ Choose correct multimodal model
MODEL_NAME = "models/gemini-2.5-flash"

cap = cv2.VideoCapture(0)  # 0 = laptop webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ✅ Generate response
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            "Describe what is happening in this live scene.",
            img
        ]
    )

    print("🧠 Gemini:", response.text)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(2)  # adjust as needed

cap.release()
cv2.destroyAllWindows()