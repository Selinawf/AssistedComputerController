from pathlib import Path
import time

import cv2
import mediapipe as mp

from controls.mouse_actions import MouseActions
from facial_recognition.features.tongue_detection import TongueDetector
from facial_recognition.features.eyebrow_highlight import EyebrowHighlighter

# SETTINGS
ENABLE_CLICKS = False

MODEL_PATH = Path(__file__).resolve().parent.parent / "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    # Draw all facial landmarks
    for point in landmarks:
        x = int(point.x * width)
        y = int(point.y * height)

        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Draw eye outlines
    eyes = [[33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]]

    for eye in eyes:
        for i in range(len(eye)):
            a = landmarks[eye[i]]
            b = landmarks[eye[(i + 1) % len(eye)]]

            p1 = (int(a.x * width), int(a.y * height))

            p2 = (int(b.x * width), int(b.y * height))

            cv2.line(frame, p1, p2, (255, 0, 0), 2)

    # Draw iris landmarks
    if len(landmarks) >= 478:
        for index in range(468, 478):
            point = landmarks[index]

            x = int(point.x * width)
            y = int(point.y * height)

            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


def draw_text(frame, message, y, color=(255, 255, 255)):
    cv2.putText(frame, message, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



def run_detector():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    # Open webcam first
    camera = cv2.VideoCapture(
        0, cv2.CAP_AVFOUNDATION
    )

    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam")

    time.sleep(2)

    # Confirm webcam works before initializing MediaPipe
    success, test_frame = camera.read()

    if not success or test_frame is None:
        camera.release()
        raise RuntimeError(
            "Camera opened but cannot capture frames"
        )

    print("Webcam initialized successfully!", flush=True)

    # Configure MediaPipe
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True
    )

    tongue = TongueDetector()
    eyebrows = EyebrowHighlighter()
    mouse = MouseActions(enabled=ENABLE_CLICKS)

    start_time = time.monotonic()
    face_visible = False


    print("AI Computer Controller started")
    print("B = baseline calibration")
    print("T = tongue calibration")
    print("Q = quit")
    print(f"Mouse enabled: {ENABLE_CLICKS}")

    try:
        with FaceLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = camera.read()

                if not success or frame is None:
                    print("Waiting for webcam...", flush=True)

                    for attempt in range(10):
                        time.sleep(0.2)
                        success, frame = camera.read()

                        if success and frame is not None:
                            print("Webcam connected!")
                            break

                if not success or frame is None:
                    print("Failed to capture frame after retries")
                    break

                frame = cv2.flip(frame, 1)
                now = time.monotonic()

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(
                    (now - start_time) * 1000
                )

                result = detector.detect_for_video(
                    mp_image, timestamp
                )

                face_visible = bool(result.face_landmarks)

                if face_visible:
                    landmarks = result.face_landmarks[0]

                    shapes = {}

                    if result.face_blendshapes:
                        shapes = {
                            item.category_name: item.score
                            for item in result.face_blendshapes[0]
                        }

                    # Tongue detection
                    score, mouth_box = tongue.calculate_score(
                        frame, landmarks
                    )

                    jaw_open = shapes.get("jawOpen", 0.0)

                    # Eyebrow highlighting
                    eyebrow_action = eyebrows.update(
                        shapes, now
                    )

                    if eyebrow_action == "start":
                        mouse.start_highlight()

                    elif eyebrow_action == "stop":
                        mouse.stop_highlight()

                    # Tongue-controlled right click
                    right_click = tongue.update(
                        score, jaw_open, now
                    )

                    if right_click and not eyebrows.highlighting:
                        mouse.right_click()

                    # Draw facial landmarks
                    draw_landmarks(frame, landmarks)

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

                    if eyebrows.highlighting:
                        draw_text(
                            frame,
                            "HIGHLIGHTING",
                            110,
                            (0, 255, 0)
                        )
                    else:
                        draw_text(
                            frame,
                            "Highlight OFF",
                            110
                        )

                    if tongue.detected:
                        draw_text(
                            frame,
                            "TONGUE DETECTED",
                            145,
                            (0, 255, 255)
                        )

                else:
                    tongue.reset()

                    if eyebrows.reset() == "stop":
                        mouse.stop_highlight()

                    draw_text(
                        frame,
                        "No face detected",
                        40,
                        (0, 0, 255)
                    )

                height = frame.shape[0]

                draw_text(
                    frame,
                    "B: Baseline  T: Tongue  Q: Quit",
                    height - 20
                )

                cv2.imshow(
                    "AI Assisted Computer Controller",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("b"):
                    if face_visible:
                        tongue.calibrate_baseline()
                    else:
                        print("No face detected")

                elif key == ord("t"):
                    if face_visible:
                        tongue.calibrate_tongue()
                    else:
                        print("No face detected")
                        
                elif key == ord("m"):
                    mouse.set_enabled(not mouse.enabled)

                elif key == ord("q"):
                    break


    finally:
        mouse.cleanup()
        camera.release()
        cv2.destroyAllWindows()

        print("AI Computer Controller stopped")