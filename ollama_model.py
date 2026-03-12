## when we press a key it gives the user description 

import cv2
import ollama_model

MODEL_NAME = "llava"
prompt = "Describe what you see in this camera image."

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # show live camera
    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1)

    # Press S to analyze frame
    if key == ord('s'):
        cv2.imwrite("frame_temp.jpg", frame)

        response = ollama_model.chat(
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

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()