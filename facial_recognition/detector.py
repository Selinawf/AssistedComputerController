
from pathlib import Path
import time

import cv2
import mediapipe as mp

from controls.mouse_actions import MouseActions
from facial_recognition.calibration import AutoCalibrator
from facial_recognition.features.tongue_detection import TongueDetector
from facial_recognition.features.eyebrow_highlight import EyebrowHighlighter
from voice_control.listener import VoiceListener
from voice_control.speech_typing import SpeechTyper

# ==========================================
# SETTINGS
# ==========================================

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
# DRAW TEXT
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
        0.55,
        color,
        2
    )


# ==========================================
# MAIN DETECTOR
# ==========================================

def run_detector():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open MacBook webcam"
        )

    # --------------------------------------
    # FEATURE MODULES
    # --------------------------------------

    tongue = TongueDetector()
    eyebrows = EyebrowHighlighter()
    mouse = MouseActions(enabled=False)

    calibrator = AutoCalibrator(tongue)
    voice = VoiceListener()

    calibrating = False
    waiting_for_release = True
    speech_typer = SpeechTyper()

    status = "Say 'Computer start calibration'"

    # --------------------------------------
    # DISABLE MOUSE
    # --------------------------------------

    def disable_mouse():
        nonlocal waiting_for_release, status

        mouse.set_enabled(False)

        eyebrows.reset()
        tongue.reset()

        waiting_for_release = True
        status = "Mouse OFF"

    # --------------------------------------
    # START CALIBRATION
    # --------------------------------------

    def begin_calibration():
        nonlocal calibrating, status

        disable_mouse()

        calibrator.reset()
        calibrating = True

        status = "Calibrating..."

        print(
            "CALIBRATION STARTED",
            flush=True
        )

    # --------------------------------------
    # ENABLE MOUSE
    # --------------------------------------

    def enable_mouse():
        nonlocal waiting_for_release, status

        if calibrating or not calibrator.finished:
            status = (
                "Complete calibration first"
            )

            print(status, flush=True)
            return

        # Release previous button state first
        mouse.stop_highlight()

        eyebrows.reset()
        tongue.reset()

        waiting_for_release = True

        try:
            mouse.set_enabled(True)
            status = (
                "Mouse ON - relax gestures first"
            )

        except Exception as error:
            disable_mouse()

            print(
                "Mouse error:",
                error,
                flush=True
            )

    # ======================================
    # START APPLICATION
    # ======================================

    try:
        time.sleep(2)

        success, test_frame = camera.read()

        if not success or test_frame is None:
            raise RuntimeError(
                "Cannot capture webcam frames"
            )

        print(
            "Webcam initialized!",
            flush=True
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True
        )

        start_time = time.monotonic()

        print(
            "Starting facial controller...",
            flush=True
        )

        # ----------------------------------
        # START MEDIAPIPE FIRST
        # ----------------------------------

        with FaceLandmarker.create_from_options(
            options
        ) as detector:

            print(
                "MediaPipe ready!",
                flush=True
            )

            # Start the separate voice process
            voice.start()

            print(
                "Voice commands:",
                flush=True
            )
            print(
                "Computer start calibration",
                flush=True
            )
            print(
                "Computer enable mouse",
                flush=True
            )
            print(
                "Computer disable mouse",
                flush=True
            )
            print(
                "Keyboard: C / M / X / Q",
                flush=True
            )

            # ==============================
            # MAIN CAMERA LOOP
            # ==============================

            while True:

                # --------------------------
                # PROCESS VOICE COMMANDS
                # --------------------------

                
                # ----------------------------------
                # VOICE COMMANDS AND DICTATION
                # ----------------------------------

                for command in voice.get_commands():

                    # Recognized speech to type
                    if isinstance(command, tuple):

                        action, text = command

                        if action == "type_text":
                            speech_typer.type_text(text)

                        continue

                    if command == "calibrate":
                        begin_calibration()

                    elif command == "enable_mouse":
                        enable_mouse()

                    elif command == "disable_mouse":
                        disable_mouse()
                        status = "Mouse OFF"

                    elif command == "start_typing":

                        # Stop facial mouse actions
                        # to prevent accidental clicks
                        disable_mouse()

                        speech_typer.start()

                        status = (
                            "DICTATION ON - speak to type"
                        )

                    elif command == "stop_typing":

                        speech_typer.stop()

                        status = "DICTATION OFF"

                    elif command == "new_line":

                        speech_typer.new_line()


                # --------------------------
                # READ CAMERA
                # --------------------------

                success, frame = camera.read()

                if not success or frame is None:
                    print(
                        "Camera frame unavailable",
                        flush=True
                    )
                    break

                frame = cv2.flip(frame, 1)

                now = time.monotonic()

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(
                    (now - start_time) * 1000
                )

                result = detector.detect_for_video(
                    image,
                    timestamp
                )

                face_visible = bool(
                    result.face_landmarks
                )

                # ==========================
                # FACE DETECTED
                # ==========================

                if face_visible:

                    landmarks = (
                        result.face_landmarks[0]
                    )

                    shapes = {}

                    if result.face_blendshapes:
                        shapes = {
                            item.category_name: item.score
                            for item in result.face_blendshapes[0]
                        }

                    score, mouth_box = (
                        tongue.calculate_score(
                            frame,
                            landmarks
                        )
                    )

                    jaw_open = shapes.get(
                        "jawOpen",
                        0.0
                    )

                    brow_score = shapes.get(
                        "browInnerUp",
                        0.0
                    )

                    # ----------------------
                    # DRAW LANDMARKS
                    # ----------------------

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

                    # ----------------------
                    # CALIBRATION
                    # ----------------------

                    if calibrating:

                        finished = calibrator.update(
                            score,
                            True,
                            frame
                        )

                        if finished:
                            calibrating = False
                            waiting_for_release = True

                            tongue.reset()
                            eyebrows.reset()

                            status = (
                                "Calibration complete! "
                                "Enable mouse when ready"
                            )

                            print(
                                "CALIBRATION COMPLETE",
                                flush=True
                            )

                    # ----------------------
                    # GESTURE CONTROLS
                    # ----------------------

                    elif (
                        mouse.enabled
                        and calibrator.finished
                    ):

                        tongue_visible = (
                            tongue.threshold is not None
                            and score > tongue.threshold
                            and jaw_open > 0.12
                        )

                        brows_neutral = (
                            brow_score
                            < eyebrows.release_threshold
                        )

                        if waiting_for_release:

                            # Do not trigger a click
                            # immediately after enabling
                            if (
                                not tongue_visible
                                and brows_neutral
                            ):
                                waiting_for_release = False

                                tongue.reset()
                                eyebrows.reset()

                                print(
                                    "GESTURES READY",
                                    flush=True
                                )

                        else:

                            # Eyebrow highlighting
                            eyebrow_action = (
                                eyebrows.update(
                                    shapes,
                                    now
                                )
                            )

                            if eyebrow_action == "start":
                                mouse.start_highlight()

                            elif eyebrow_action == "stop":
                                mouse.stop_highlight()

                            # Tongue right-click
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

                    # ----------------------
                    # CONTROLS INACTIVE
                    # ----------------------

                    else:
                        tongue.reset()
                        eyebrows.reset()
                        mouse.stop_highlight()

                    # ----------------------
                    # DISPLAY SCORES
                    # ----------------------

                    draw_text(
                        frame,
                        f"Tongue score: {score:.1f}",
                        40
                    )

                    draw_text(
                        frame,
                        f"Eyebrows: {brow_score:.2f}",
                        70
                    )

                    if mouse.dragging:
                        draw_text(
                            frame,
                            "HIGHLIGHTING",
                            100,
                            (0, 255, 0)
                        )

                # ==========================
                # FACE LOST
                # ==========================

                else:
                    tongue.reset()
                    eyebrows.reset()

                    mouse.stop_highlight()

                    waiting_for_release = True

                    if calibrating:
                        calibrator.update(
                            None,
                            False,
                            frame
                        )

                    draw_text(
                        frame,
                        "No face detected",
                        40,
                        (0, 0, 255)
                    )

                # ==========================
                # STATUS DISPLAY
                # ==========================

                height = frame.shape[0]

                mouse_label = (
                    "ON" if mouse.enabled
                    else "OFF"
                )

                if voice.ready:
                    voice_label = "READY"

                elif voice.error:
                    voice_label = "ERROR"

                else:
                    voice_label = "STARTING"

                draw_text(
                    frame,
                    (
                        f"Mouse: {mouse_label} "
                        f"| Voice: {voice_label}"
                    ),
                    height - 90
                )

                draw_text(
                    frame,
                    status[:60],
                    height - 60,
                    (0, 255, 255)
                )

                draw_text(
                    frame,
                    (
                        "C: Calibrate M: Enable "
                        "X: Disable Q: Quit"
                    ),
                    height - 25
                )

                cv2.imshow(
                    "AI Assisted Computer Controller",
                    frame
                )

                # ==========================
                # KEYBOARD BACKUPS
                # ==========================

                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):
                    begin_calibration()

                elif key == ord("m"):
                    enable_mouse()

                elif key == ord("x"):
                    disable_mouse()

                elif key == ord("q"):
                    break

    # ======================================
    # CLEANUP
    # ======================================

    except KeyboardInterrupt:
        print(
            "Stopped by user",
            flush=True
        )

    finally:
        speech_typer.stop()
        mouse.cleanup()
        voice.stop()

        camera.release()
        cv2.destroyAllWindows()

        print(
            "AI Computer Controller stopped",
            flush=True
        )
