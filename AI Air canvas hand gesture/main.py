import cv2
import numpy as np
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = None, None
draw_color = (0, 255, 0)
mode = "DRAW"

BAR_HEIGHT = 50

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # 🎨 Top bar
    cv2.rectangle(frame, (0, 0), (128, BAR_HEIGHT), (128, 0, 128), -1)   # Purple
    cv2.rectangle(frame, (128, 0), (256, BAR_HEIGHT), (255, 0, 0), -1)   # Blue
    cv2.rectangle(frame, (256, 0), (384, BAR_HEIGHT), (0, 255, 0), -1)   # Green
    cv2.rectangle(frame, (384, 0), (512, BAR_HEIGHT), (0, 255, 255), -1) # Yellow
    cv2.rectangle(frame, (512, 0), (640, BAR_HEIGHT), (50, 50, 50), -1)  # Eraser

    cv2.putText(frame, "ERASER", (525, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * w), int(index_finger.y * h)

            # 🎨 Selection zone
            if y < BAR_HEIGHT:
                if 0 < x < 128:
                    draw_color = (128, 0, 128)
                    mode = "DRAW"
                elif 128 < x < 256:
                    draw_color = (255, 0, 0)
                    mode = "DRAW"
                elif 256 < x < 384:
                    draw_color = (0, 255, 0)
                    mode = "DRAW"
                elif 384 < x < 512:
                    draw_color = (0, 255, 255)
                    mode = "DRAW"
                elif 512 < x < 640:
                    mode = "ERASE"

                prev_x, prev_y = None, None

            else:
                if prev_x is None:
                    prev_x, prev_y = x, y

                if mode == "DRAW":
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 5)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 30)

                prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    frame = cv2.add(frame, canvas)
    cv2.imshow("AI Air Canvas", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
