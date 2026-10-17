
import time

import cv2
import mediapipe as mp
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from eye_tracking.eye_blinking import BlinkDetector

# Configuration
MODEL_PATH = "face_landmarker.task"
CAMERA_ID = 0
SENSITIVITY = 2.5
DEAD_ZONE = 0.025
SMOOTHING = 0.25
NOSE_TIP = 1

RANGE_X = 0.15
RANGE_Y = 0.12

pyautogui.FAILSAFE = True
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
def headinitialize():
    calibration_x = None
    calibration_y = None

    current_x = screen_w / 2
    current_y = screen_h / 2

start_time = time.monotonic()

try:
    with vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Mirror the preview and tracking direction
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            h, w = frame.shape[:2]

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                nose = landmarks[NOSE_TIP]

                nx, ny = nose.x, nose.y

                # Initialize neutral position
                if calibration_x is None:
                    calibration_x = nx
                    calibration_y = ny

                dx = nx - calibration_x
                dy = ny - calibration_y

                # Ignore tiny movements caused by tracking jitter
                if abs(dx) < DEAD_ZONE:
                    dx = 0.0
                else:
                    dx = (
                        dx - DEAD_ZONE * (1 if dx > 0 else -1)
                    )

                if abs(dy) < DEAD_ZONE:
                    dy = 0.0
                else:
                    dy = (
                        dy - DEAD_ZONE * (1 if dy > 0 else -1)
                    )

                # Map head movement to screen coordinates
                target_x = screen_w / 2 + (dx / RANGE_X) * (screen_w / 2)
                target_y = screen_h / 2 + (dy / RANGE_Y) * (screen_h / 2)

                target_x = max(1, min(screen_w - 2, target_x))
                target_y = max(1, min(screen_h - 2, target_y))

                # Smooth the cursor
                current_x += (target_x - current_x) * SMOOTHING
                current_y += (target_y - current_y) * SMOOTHING

                pyautogui.moveTo(int(current_x), int(current_y))

                pyautogui.moveTo(
                    int(current_x), int(current_y)
                )

                px, py = int(nx * w), int(ny * h)

                cv2.circle(
                    frame, (px, py), 8, (0, 255, 0), -1
                )

                cv2.putText(
                    frame,
                    "Nose tracking active",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    frame,
                    "Face not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )

            cv2.putText(
                frame,
                "C: recalibrate | Q: quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.imshow("MediaPipe Nose Cursor", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if result.face_landmarks:
                    nose = result.face_landmarks[0][NOSE_TIP]
                    calibration_x = nose.x
                    calibration_y = nose.y

            elif key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
