
import cv2
import mediapipe as mp
import pyautogui
import time
import math
from pathlib import Path

# ================= SETTINGS =================

SCROLL_MAX = 30
TILT_THRESHOLD = 0.025
SCROLL_SENSITIVITY = 600
TILT_X_THRESHOLD = 3
SMOOTHING = 0.25
TAB_COOLDOWN = 1.0
RECALIBRATION_DELAY = 3.0

pyautogui.FAILSAFE = True

MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

# ================= MEDIAPIPE =================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
Options = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = Options(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# ================= STATE =================

neutral_y = None
neutral_x = None

smooth_tilt = 0.0
smooth_x = 0.0

recalibrate_time = None
last_tab_switch = 0.0
tab_tilt_active = False

start_time = time.monotonic()
last_timestamp_ms = -1

print("Look straight ahead to calibrate.")
print("UP/DOWN = scroll")
print("LEFT/RIGHT = switch tabs")
print("R = recalibrate in 3 seconds")
print("Q = quit")

try:
    with FaceLandmarker.create_from_options(options) as tracker:

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Mirror the preview
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            result = tracker.detect_for_video(
                mp_image, timestamp_ms
            )

            status = "No face detected"
            now = time.monotonic()

            if result.face_landmarks:
                face = result.face_landmarks[0]

                forehead = face[10]
                nose = face[1]
                chin = face[152]
                left_eye = face[33]
                right_eye = face[263]

                face_height = chin.y - forehead.y

                eye_mid_y = (
                    left_eye.y + right_eye.y
                ) / 2

                face_width = abs(
                    right_eye.x - left_eye.x
                )

                if face_height > 0 and face_width > 0:

                    # Vertical movement: head up/down
                    tilt_y = (
                        nose.y - eye_mid_y
                    ) / face_height

                    # Sideways head roll: angle of eye line
                    eye_angle = math.degrees(math.atan2(
                    right_eye.y - left_eye.y,
                    right_eye.x - left_eye.x
                    ))

                    tilt_x = eye_angle

                    # Initial calibration
                    if neutral_y is None:
                        neutral_y = tilt_y
                        neutral_x = tilt_x
                        smooth_tilt = 0.0
                        smooth_x = 0.0

                        status = "Calibrated - neutral"

                    # Recalibration countdown
                    elif recalibrate_time is not None:
                        elapsed = now - recalibrate_time
                        remaining = max(
                            0,
                            math.ceil(
                                RECALIBRATION_DELAY - elapsed
                            )
                        )

                        if elapsed >= RECALIBRATION_DELAY:
                            neutral_y = tilt_y
                            neutral_x = tilt_x

                            smooth_tilt = 0.0
                            smooth_x = 0.0

                            recalibrate_time = None
                            tab_tilt_active = False

                            status = "Recalibration complete"

                        else:
                            status = (
                                f"Recalibrating in {remaining}..."
                            )

                    # Only control scrolling/tabs outside countdown
                    else:
                        smooth_tilt += SMOOTHING * (
                            (tilt_y - neutral_y)
                            - smooth_tilt
                        )

                        smooth_x += SMOOTHING * (
                            (tilt_x - neutral_x)
                            - smooth_x
                        )

                        
                    # -------- SCROLLING AND TAB SWITCHING --------

                    # Determine whether the head is currently being used to scroll
                    is_scrolling = abs(smooth_tilt) > TILT_THRESHOLD

                    if is_scrolling:
                        # Disable tab switching whenever scrolling is active
                        tab_tilt_active = False

                        if smooth_tilt > TILT_THRESHOLD:
                            # Head down = scroll down
                            tilt_strength = smooth_tilt - TILT_THRESHOLD

                            scroll_amount = int(min(
                                SCROLL_MAX,
                                1 + tilt_strength * SCROLL_SENSITIVITY
                            ))

                            pyautogui.scroll(-int(scroll_amount))
                            status = f"SCROLL DOWN ({scroll_amount})"

                        else:
                            # Head up = scroll up
                            tilt_strength = abs(smooth_tilt) - TILT_THRESHOLD

                            scroll_amount = int(min(
                                SCROLL_MAX,
                                1 + tilt_strength * SCROLL_SENSITIVITY
                            ))

                            pyautogui.scroll(int(scroll_amount))
                            status = f"SCROLL UP ({scroll_amount})"

                    else:
                        # Only allow tab switching when not scrolling

                        if abs(smooth_x) < TILT_X_THRESHOLD * 0.5:
                            tab_tilt_active = False

                        if not tab_tilt_active:
                            if now - last_tab_switch >= TAB_COOLDOWN:

                                if smooth_x <= -TILT_X_THRESHOLD:
                                    pyautogui.hotkey("ctrl", "shift", "tab")
                                    last_tab_switch = now
                                    tab_tilt_active = True
                                    status = "LEFT: previous tab"

                                elif smooth_x >= TILT_X_THRESHOLD:
                                    pyautogui.hotkey("ctrl", "tab")
                                    last_tab_switch = now
                                    tab_tilt_active = True
                                    status = "RIGHT: next tab"

                        if not tab_tilt_active:
                            status = "NEUTRAL: stopped"
                    print(
                    f"Vertical: {smooth_tilt:.3f}, "
                    f"Horizontal: {smooth_x:.3f}"
                    )

                    # Debug information
                    cv2.putText(
                        frame,
                        f"Vertical: {smooth_tilt:.3f}",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Horizontal: {smooth_x:.3f}",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

            # Show status
            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            # Show recalibration countdown
            if recalibrate_time is not None:
                remaining = max(
                    0,
                    math.ceil(
                        RECALIBRATION_DELAY
                        - (now - recalibrate_time)
                    )
                )

                cv2.putText(
                    frame,
                    f"Recalibrating in {remaining}...",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )

            cv2.imshow("Head Tracking Scroll", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                recalibrate_time = time.monotonic()
                smooth_tilt = 0.0
                smooth_x = 0.0
                tab_tilt_active = False

                print("Recalibrating in 3 seconds. Look straight ahead!")

finally:
    cap.release()
    cv2.destroyAllWindows()
