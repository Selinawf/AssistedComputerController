
from pathlib import Path
import time

import cv2
import mediapipe as mp
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from controls.mouse_actions import MouseActions
from facial_recognition.calibration import AutoCalibrator
from facial_recognition.features.tongue_detection import TongueDetector
from facial_recognition.features.eyebrow_highlight import EyebrowHighlighter


# ==========================================
# SETTINGS
# ==========================================

ENABLE_CLICKS = False

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "face_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# ==========================================
# DRAW FACIAL LANDMARKS
# ==========================================

def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    # Draw all facial landmarks
    for point in landmarks:
        x = int(point.x * width)
        y = int(point.y * height)

        cv2.circle(
            frame,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    # Draw eye outlines
    eyes = [
        [33, 160, 158, 133, 153, 144],
        [362, 385, 387, 263, 373, 380]
    ]

    for eye in eyes:
        for i in range(len(eye)):
            a = landmarks[eye[i]]
            b = landmarks[eye[(i + 1) % len(eye)]]

            p1 = (
                int(a.x * width),
                int(a.y * height)
            )

            p2 = (
                int(b.x * width),
                int(b.y * height)
            )

            cv2.line(
                frame,
                p1,
                p2,
                (255, 0, 0),
                2
            )

    # Draw iris landmarks
    if len(landmarks) >= 478:
        for index in range(468, 478):
            point = landmarks[index]

            x = int(point.x * width)
            y = int(point.y * height)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 255),
                -1
            )


# ==========================================
# DRAW TEXT ON WEBCAM
# ==========================================

def draw_text(
    frame,
    message,
    y,
    color=(255, 255, 255)
):
    cv2.putText(
        frame,
        message,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


# ==========================================
# MAIN FACIAL DETECTOR
# ==========================================

def run_detector():

    # Verify MediaPipe model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    # --------------------------------------
    # INITIALIZE CAMERA FIRST
    # --------------------------------------
    pyautogui.FAILSAFE = True
    screen_w, screen_h = pyautogui.size()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam")

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    # --------------------------------------
    # CONFIGURE MEDIAPIPE
    # --------------------------------------

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True
    )

    # --------------------------------------
    # INITIALIZE FEATURE MODULES
    # --------------------------------------

    tongue = TongueDetector()
    eyebrows = EyebrowHighlighter()
    mouse = MouseActions(enabled=ENABLE_CLICKS)

    # Automatic tongue calibration
    calibrator = AutoCalibrator(tongue)

    # Prevent an immediate click when
    # automatic calibration finishes
    waiting_for_release = True

    start_time = time.monotonic()

    print("AI Computer Controller started")
    print("Automatic tongue calibration enabled")
    print("M = toggle mouse controls")
    print("Q = quit")
    print(f"Mouse enabled: {ENABLE_CLICKS}")

    # --------------------------------------
    # MAIN CAMERA LOOP
    # --------------------------------------

    try:
        with FaceLandmarker.create_from_options(
            options
        ) as detector:

            while True:

                success, frame = camera.read()

                # Retry if the webcam temporarily fails
                if not success or frame is None:
                    print(
                        "Waiting for webcam...",
                        flush=True
                    )

                    for _ in range(10):
                        time.sleep(0.2)

                        success, frame = camera.read()

                        if success and frame is not None:
                            print(
                                "Webcam connected!",
                                flush=True
                            )
                            break

                if not success or frame is None:
                    print(
                        "Failed to capture frame after retries"
                    )
                    break

                # Mirror webcam image
                frame = cv2.flip(frame, 1)

                now = time.monotonic()

                # ----------------------------------
                # MEDIAPIPE FACE DETECTION
                # ----------------------------------

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(
                    (now - start_time) * 1000
                )

                result = detector.detect_for_video(
                    mp_image,
                    timestamp
                )

                face_visible = bool(
                    result.face_landmarks
                )

                # ----------------------------------
                # PROCESS FACIAL FEATURES
                # ----------------------------------

                if face_visible:

                    landmarks = result.face_landmarks[0]

                    # Get facial blendshape scores
                    shapes = {}

                    if result.face_blendshapes:
                        shapes = {
                            item.category_name: item.score
                            for item in result.face_blendshapes[0]
                        }

                    jaw_open = shapes.get(
                        "jawOpen",
                        0.0
                    )

                    # ----------------------------------
                    # TONGUE DETECTION SCORE
                    # ----------------------------------

                    score, mouth_box = (
                        tongue.calculate_score(
                            frame,
                            landmarks
                        )
                    )

                    # ----------------------------------
                    # AUTOMATIC CALIBRATION
                    # ----------------------------------

                    was_calibrated = calibrator.finished

                    calibrated = calibrator.update(
                        score,
                        True,
                        frame
                    )

                    if not was_calibrated and calibrated:
                        waiting_for_release = True

                    if not calibrated:

                        # Disable gesture actions while
                        # calibration is running
                        tongue.reset()
                        eyebrows.reset()
                        mouse.stop_highlight()

                    else:

                        # ----------------------------------
                        # WAIT FOR TONGUE RELEASE
                        # ----------------------------------

                        tongue_visible = (
                            tongue.threshold is not None
                            and score > tongue.threshold
                            and jaw_open > 0.12
                        )

                        if waiting_for_release:

                            if not tongue_visible:
                                waiting_for_release = False

                                print(
                                    "Gesture controls ready!"
                                )

                        # ----------------------------------
                        # PROCESS GESTURES
                        # ----------------------------------

                        if not waiting_for_release:

                            # Eyebrow highlighting
                            eyebrow_action = eyebrows.update(
                                shapes,
                                now
                            )

                            if eyebrow_action == "start":
                                mouse.start_highlight()

                            elif eyebrow_action == "stop":
                                mouse.stop_highlight()

                            # Tongue-controlled right click
                            right_click = tongue.update(
                                score,
                                jaw_open,
                                now
                            )

                            if (
                                right_click
                                and not eyebrows.highlighting
                            ):
                                mouse.right_click()

                    # ----------------------------------
                    # DRAW VISUALIZATION
                    # ----------------------------------

                    draw_landmarks(
                        frame,
                        landmarks
                    )

                    if mouth_box:
                        x1, y1, x2, y2 = mouth_box

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 255, 0),
                            2
                        )

                    draw_text(
                        frame,
                        f"Tongue score: {score:.3f}",
                        40
                    )

                    draw_text(
                        frame,
                        f"Eyebrow score: {eyebrows.score:.2f}",
                        75
                    )

                    # ----------------------------------
                    # STATUS DISPLAY
                    # ----------------------------------

                    if not calibrated:

                        draw_text(
                            frame,
                            "CALIBRATING...",
                            110,
                            (0, 255, 255)
                        )

                    elif waiting_for_release:

                        draw_text(
                            frame,
                            "Retract tongue to begin",
                            110,
                            (0, 255, 255)
                        )

                    elif eyebrows.highlighting:

                        draw_text(
                            frame,
                            "HIGHLIGHTING",
                            110,
                            (0, 255, 0)
                        )

                    else:

                        draw_text(
                            frame,
                            "Controls Ready",
                            110,
                            (0, 255, 0)
                        )

                    if calibrated and tongue.detected:

                        draw_text(
                            frame,
                            "TONGUE DETECTED",
                            145,
                            (0, 255, 255)
                        )

                else:

                    # ----------------------------------
                    # FACE LOST - SAFETY RESET
                    # ----------------------------------

                    tongue.reset()

                    if eyebrows.reset() == "stop":
                        mouse.stop_highlight()

                    # Return calibration to ready stage
                    # if calibration has not finished
                    if not calibrator.finished:
                        calibrator.update(
                            None,
                            False,
                            frame
                        )

                    waiting_for_release = True

                    draw_text(
                        frame,
                        "No face detected",
                        40,
                        (0, 0, 255)
                    )

                # ----------------------------------
                # DISPLAY MOUSE STATUS
                # ----------------------------------

                height = frame.shape[0]

                mouse_status = (
                    "ON" if mouse.enabled else "OFF"
                )

                draw_text(
                    frame,
                    f"Mouse Controls: {mouse_status}",
                    height - 55,
                    (0, 255, 0)
                    if mouse.enabled
                    else (0, 0, 255)
                )

                draw_text(
                    frame,
                    "M: Toggle Mouse  Q: Quit",
                    height - 20
                )

                cv2.imshow(
                    "AI Assisted Computer Controller",
                    frame
                )

                # ----------------------------------
                # KEYBOARD SHORTCUTS
                # ----------------------------------

                key = cv2.waitKey(1) & 0xFF

                if key == ord("m"):

                    # Do not allow real mouse actions
                    # before calibration succeeds
                    if calibrator.finished:
                        mouse.set_enabled(
                            not mouse.enabled
                        )
                    else:
                        print(
                            "Complete calibration first."
                        )

                elif key == ord("q"):
                    break

    # --------------------------------------
    # CLEANUP
    # --------------------------------------

    finally:
        mouse.cleanup()
        camera.release()
        cv2.destroyAllWindows()

        print(
            "AI Computer Controller stopped"
        )
