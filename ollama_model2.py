import cv2
import ollama
import time

MODEL_NAME = "llava"
prompt = "Describe this camera image in maximum two short lines."

last_analysis = 0
interval = 5

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    current_time = time.time()

    # run every 5 seconds
    if current_time - last_analysis >= interval:

        print("Analyzing frame...")

        cv2.imwrite("frame_temp.jpg", frame)

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": ["frame_temp.jpg"]
                }
            ]
        )

        text = response["message"]["content"].strip()
        print("AI:", text)

        last_analysis = current_time

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()