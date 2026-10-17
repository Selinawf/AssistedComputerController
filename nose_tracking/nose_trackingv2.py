import cv2
import mediapipe as mp
import pyautogui
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"

# Adjust these settings
DEAD_ZONE_X = 1.5       # Minimum left/right tilt, degrees
DEAD_ZONE_Y = 0.02     # Minimum up/down movement
MAX_TILT_X = 10.0       # Tilt needed for maximum horizontal speed
MAX_TILT_Y = 0.12       # Normalized nod needed for max vertical speed
SPEED_X = 25.0          # Horizontal speed in pixels per frame
SPEED_Y = 20.0          # Vertical speed in pixels per frame
SMOOTHING = 0.25

pyautogui.FAILSAFE = True
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

neutral_roll = None
neutral_pitch = None

cursor_x, cursor_y = map(float, pyautogui.position())
velocity_x = 0.0
velocity_y = 0.0

start = time.monotonic()
last_timestamp = -1


def wrap_angle(angle):
    """Keep angular difference within -180 to 180 degrees."""
    return (angle + 180) % 360 - 180


def response(value, dead_zone, max_value):
    """Convert tilt into a smooth signed movement amount."""
    magnitude = abs(value)

    if magnitude <= dead_zone:
        return 0.0

    amount = min(
        (magnitude - dead_zone) / (max_value - dead_zone),
        1.0
    )
    return math.copysign(amount, value)


try:
    with vision.FaceLandmarker.create_from_options(options) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int((time.monotonic() - start) * 1000)
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = tracker.detect_for_video(image, timestamp)

            roll = None
            pitch = None

            if result.face_landmarks:
                lm = result.face_landmarks[0]

                # Eye corners and nose tip
                eye_a = (lm[33].x * w, lm[33].y * h)
                eye_b = (lm[263].x * w, lm[263].y * h)
                nose = (lm[1].x * w, lm[1].y * h)
                chin = (lm[152].x * w, lm[152].y * h)

                eye_mid_x = (eye_a[0] + eye_b[0]) / 2
                eye_mid_y = (eye_a[1] + eye_b[1]) / 2

                eye_distance = math.dist(eye_a, eye_b)
                face_height = math.dist(
                    (eye_mid_x, eye_mid_y), chin
                )

                # Head roll: angle of the eye line
                roll = math.degrees(math.atan2(
                    eye_b[1] - eye_a[1],
                    eye_b[0] - eye_a[0]
                ))

                # Head pitch proxy: nose position relative to eyes,
                # normalized by face size to reduce distance effects.
                if face_height > 1:
                    pitch = (
                        (nose[1] - eye_mid_y) / face_height
                    )

                if neutral_roll is None:
                    neutral_roll = roll
                    neutral_pitch = pitch

                tilt_x = wrap_angle(roll - neutral_roll)
                tilt_y = pitch - neutral_pitch

                # Tilt controls movement speed.
                move_x = response(
                    tilt_x, DEAD_ZONE_X, MAX_TILT_X
                ) * SPEED_X

                move_y = response(
                    tilt_y, DEAD_ZONE_Y, MAX_TILT_Y
                ) * SPEED_Y

                velocity_x += (
                    move_x - velocity_x
                ) * SMOOTHING

                velocity_y += (
                    move_y - velocity_y
                ) * SMOOTHING

                # Keep floating-point position to preserve tiny movements
                cursor_x = max(
                    0, min(screen_w - 1, cursor_x + velocity_x)
                )
                cursor_y = max(
                    0, min(screen_h - 1, cursor_y + velocity_y)
                )

                pyautogui.moveTo(
                    round(cursor_x), round(cursor_y)
                )

                cv2.circle(
                    frame,
                    (int(nose[0]), int(nose[1])),
                    6, (0, 255, 0), -1
                )

                cv2.putText(
                    frame, f"Roll: {tilt_x:+.1f} deg",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2
                )

                cv2.putText(
                    frame, f"Pitch change: {tilt_y:+.3f}",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Velocity: {velocity_x:+.1f}, {velocity_y:+.1f}",
                    (15, 95), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 0), 2
                )

            else:
                # Stop cursor when the face is lost
                velocity_x *= 0.5
                velocity_y *= 0.5

            cv2.putText(
                frame, "C: calibrate | Q: quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.imshow("Head Tilt Mouse", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c") and roll is not None:
                neutral_roll = roll
                neutral_pitch = pitch
                velocity_x = velocity_y = 0.0
            elif key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
