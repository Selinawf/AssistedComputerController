
from pathlib import Path
import time

import cv2
import mediapipe as mp
import pyautogui


from voice_control.voiceassistant import VoiceAssistant

from controls.mouse_actions import MouseActions
from facial_recognition.calibration import AutoCalibrator
from facial_recognition.features.tongue_detection import TongueDetector
from facial_recognition.features.eyebrow_highlight import EyebrowHighlighter
from eye_tracking.eye_blinking import BlinkDetector
from voice_control.listener import VoiceListener
from voice_control.speech_typing import SpeechTyper
from nose_tracking.nose_bridge import NoseCursor
from controls.command_menu import draw_command_menu

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "face_landmarker.task"
)

pyautogui.FAILSAFE = True

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# ==========================================
# DISPLAY HELPERS
# ==========================================

def draw_text(frame, message, y, color=(255, 255, 255)):
    cv2.putText(
        frame,
        str(message),
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    for point in landmarks:
        x = int(point.x * width)
        y = int(point.y * height)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    eyes = [
        [33, 160, 158, 133, 153, 144],
        [362, 385, 387, 263, 373, 380],
    ]

    
    eyes: list[list[int]] = [
        [33, 160, 158, 133, 153, 144],
        [362, 385, 387, 263, 373, 380],
    ]

    for eye in eyes:
        for i in range(len(eye)):
            a = landmarks[eye[i]]
            b = landmarks[eye[(i + 1) % len(eye)]]

            p1 = (int(a.x * width), int(a.y * height))
            p2 = (int(b.x * width), int(b.y * height))

            cv2.line(frame, p1, p2, (255, 0, 0), 2)


            p1 = (int(a.x * width), int(a.y * height))
            p2 = (int(b.x * width), int(b.y * height))

            cv2.line(frame, p1, p2, (255, 0, 0), 2)

    if len(landmarks) >= 478:
        for index in range(468, 478):
            point = landmarks[index]

            cv2.circle(
                frame,
                (int(point.x * width), int(point.y * height)),
                3,
                (0, 255, 255),
                -1,
            )


# ==========================================
# MAIN APPLICATION
# ==========================================

def run_detector():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"MediaPipe model missing: {MODEL_PATH}"
        )

    # Open the Mac webcam before MediaPipe.
    camera = cv2.VideoCapture(
        0)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Cannot open webcam")

    tongue = TongueDetector()
    eyebrows = EyebrowHighlighter()
    blink_detector = BlinkDetector()

    nose_cursor = NoseCursor()

    # VoiceListener is started once after MediaPipe initialization.
    irisvoice = VoiceAssistant()
    
    mouse = MouseActions(enabled=False)
    calibrator = AutoCalibrator(tongue)

    voice = VoiceListener()
    speech_typer = SpeechTyper()

    # Real clicking starts disabled.
    blink_detector.enabled = True

    calibrating = False
    waiting_for_release = True
    status = "Say Computer start calibration"
    menu_state: dict[str, str | None] = {"current": None}

    # ======================================
    # VOICE SCROLL SETTINGS
    # ======================================

    scroll_amount = 5
    scroll_min = 1
    scroll_max = 20

    # ======================================
    # CONTROL HELPERS
    # ======================================

    def reset_gestures():
        tongue.reset()
        eyebrows.reset()
        blink_detector.reset()
        mouse.stop_highlight()

    def disable_mouse():
        nonlocal waiting_for_release, status

        mouse.set_enabled(False)
        reset_gestures()
        waiting_for_release = True

        status = "Mouse OFF"
        print("MOUSE OFF", flush=True)

    def begin_calibration():
        nonlocal calibrating, status

        disable_mouse()
        speech_typer.stop()
        calibrator.reset()

        calibrating = True
        status = "Calibrating tongue..."

        print("CALIBRATION STARTED", flush=True)

    def enable_mouse():
        nonlocal waiting_for_release, status

        if calibrating or not calibrator.finished:
            status = "Complete calibration first"
            print(status, flush=True)
            irisvoice.speak("Complete calibration first.")
            return

        if speech_typer.enabled:
            status = "Stop dictation first"
            irisvoice.speak("Stop dictation first.")
            print(status, flush=True)
            return

        reset_gestures()
        waiting_for_release = True

        mouse.set_enabled(True)

        status = "Mouse ON - relax your face"
        print("MOUSE ON", flush=True)
        irisvoice.speak("Mouse on.")

    def set_blink(enabled):
        nonlocal status

        blink_detector.enabled = enabled
        blink_detector.reset()

        status = (
            "Double Blink ON"
            if enabled
            else "Double Blink OFF"
        )

        print(status, flush=True)

    def handle_voice_command(command):
        nonlocal status, scroll_amount

        print(
            "Received voice command:",
            command,
            flush=True,
        )

        if isinstance(command, tuple):
            action, value = command

            if action == "type_text":
                speech_typer.type_text(value)

            elif action == "delete_words":
                speech_typer.delete_word(value)

            return


        # ======================================
        # BROWSER TAB VOICE COMMANDS (macOS)
        # ======================================
        # Shortcuts apply to the active application.
        tab_shortcuts = {
            "new_tab": ("command", "t"),
            "close_tab": ("command", "w"),
            "next_tab": ("ctrl", "tab"),
            "previous_tab": ("ctrl", "shift", "tab"),
            "reopen_tab": ("command", "shift", "t"),
        }
        if command in tab_shortcuts:
            pyautogui.hotkey(*tab_shortcuts[command])
            status = command.replace("_", " ").title()
            print(status, flush=True)
            return

        # ======================================
        # VOICE SCROLL COMMANDS
        # ======================================
        if command in (
            "scroll_up",
            "scroll_down",
            "scroll_up_little",
            "scroll_down_little",
            "scroll_up_lot",
            "scroll_down_lot",
        ):
            if command.endswith("_little"):
                amount = max(1, scroll_amount // 2)

            elif command.endswith("_lot"):
                amount = scroll_amount * 3

            else:
                amount = scroll_amount

            direction = 1 if "up" in command else -1

            pyautogui.scroll(direction * amount)

            status = f"Scrolled {command.replace('_', ' ')}"
            print(status, flush=True)
            return

        elif command == "scroll_faster":
            scroll_amount = min(
                scroll_max,
                scroll_amount + 2
            )

            status = f"Scroll speed: {scroll_amount}"
            print(status, flush=True)
            return

        elif command == "scroll_slower":
            scroll_amount = max(
                scroll_min,
                scroll_amount - 2
            )

            status = f"Scroll speed: {scroll_amount}"
            print(status, flush=True)
            return

        elif command == "stop_scrolling":
            status = "Scrolling stopped"
            print(status, flush=True)
            return

        
        # ======================================
        # VOICE WINDOW SWITCHING
        # ======================================

        
        # MACOS APPLICATION SWITCHING

        # macOS app switcher follows its recent-use order, NOT
        # the left/right screen positions of application windows.
        if command in ("switch_window_right", "switch_window_left"):
            going_right = command == "switch_window_right"
            pyautogui.keyDown("command")
            try:
                if not going_right:
                    pyautogui.keyDown("shift")
                try:
                    # Open switcher then move another step.
                    # The currently selected app depends on recency.
                    pyautogui.press("tab", presses=2, interval=0.15)
                    time.sleep(0.2)
                finally:
                    if not going_right:
                        pyautogui.keyUp("shift")
            finally:
                pyautogui.keyUp("command")

            direction = "right" if going_right else "left"
            status = f"App switcher moved {direction}"
            print(status, flush=True)
            return

        if command == "calibrate":
            begin_calibration()

        elif command == "enable_mouse":
            enable_mouse()

        elif command == "disable_mouse":
            disable_mouse()

        elif command == "enable_blink":
            set_blink(True)

        elif command == "disable_blink":
            set_blink(False)


        elif command == "enable_nose":
            if not nose_cursor.enabled:
                nose_cursor.start()
            status = "Nose tracking ON"
            irisvoice.speak("Nose tracking on.")

        elif command == "disable_nose":
            if nose_cursor.enabled:
                nose_cursor.stop()
            status = "Nose tracking OFF"
            irisvoice.speak("Nose tracking off.")

        elif command == "recenter_nose":
            if nose_cursor.enabled:
                nose_cursor.recalibrate()
                status = "Nose recentered"
            else:
                status = "Enable nose tracking first"

        elif command == "freeze_cursor":
            if nose_cursor.enabled:
                nose_cursor.freeze()
                status = "Cursor frozen"
            else:
                status = "Enable nose tracking first"

        elif command == "unfreeze_cursor":
            if nose_cursor.enabled:
                nose_cursor.unfreeze()
                status = "Cursor unfrozen"
            else:
                status = "Enable nose tracking first"

        elif command == "start_typing":
            disable_mouse()

            if nose_cursor.enabled:
                nose_cursor.freeze()

            speech_typer.start()
            status = "DICTATION ON"
            irisvoice.speak("Dictation started.")

        elif command == "stop_typing":
            speech_typer.stop()
            if nose_cursor.enabled:
                nose_cursor.unfreeze()
            status = "DICTATION OFF"
            irisvoice.speak("Dictation stopped.")

        elif command == "new_line":
            speech_typer.new_line()
        elif isinstance(command, tuple) and command[0] == "delete_words":
            speech_typer.delete_word(command[1])


        if command == "show_voice_menu":
            menu_state["current"] = "voice"
            status = "Voice command menu"
            return

        elif command == "show_keyboard_menu":
            menu_state["current"] = "keyboard"
            status = "Keyboard command menu"
            return

        elif command == "hide_commands":
            menu_state["current"] = None
            status = "Commands hidden"
            return
    # ======================================
    # START THE APPLICATION
    # ======================================

    try:
        # Allow the MacBook webcam to warm up
        print("Initializing webcam...", flush=True)

        success = False
        test_frame = None

        for attempt in range(30):
            success, test_frame = camera.read()

            if success and test_frame is not None:
                print(
                    f"Webcam ready after {attempt + 1} attempts",
                    flush=True
                )
                break

            print(
                f"Waiting for webcam... {attempt + 1}/30",
                flush=True
            )

            time.sleep(0.2)

        if not success or test_frame is None:
            raise RuntimeError(
                "Webcam could not capture frames. "
                "Check camera permissions and close "
                "other applications using the camera."
            )

        print("Webcam initialized!", flush=True)

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )

        start_time = time.monotonic()
        last_timestamp = -1

        with FaceLandmarker.create_from_options(
            options
        ) as detector:

            print("MediaPipe ready!", flush=True)

            # Voice runs in a separate Python process.
            if voice.start():
                print(
                    "Voice subprocess starting...",
                    flush=True,
                )
            else:
                print(
                    "Voice startup failed:",
                    voice.error,
                    flush=True,
                )

            print(
                "C Calibrate | M Mouse ON | X Mouse OFF",
                flush=True,
            )
            print(
                "K Toggle Blink | Q Quit",
                flush=True,
            )

            # ==================================
            # MAIN CAMERA LOOP
            # ==================================

            while True:

                # ------------------------------
                # VOICE COMMANDS
                # ------------------------------

                for command in voice.get_commands():
                    handle_voice_command(command)

                # ------------------------------
                # CAMERA AND MEDIAPIPE
                # ------------------------------

                
                success, frame = camera.read()

                if not success or frame is None:
                    print(
                        "Camera frame unavailable - retrying...",
                        flush=True
                    )

                    # Release any held mouse button
                    mouse.stop_highlight()

                    # Try recovering before exiting
                    for attempt in range(10):
                        time.sleep(0.2)
                        success, frame = camera.read()

                        if success and frame is not None:
                            print(
                                "Camera recovered!",
                                flush=True
                            )
                            break

                if not success or frame is None:
                    print(
                        "Camera recovery failed.",
                        flush=True
                    )
                    break

                frame = cv2.flip(frame, 1)
                now = time.monotonic()

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp = max(
                    last_timestamp + 1,
                    int((now - start_time) * 1000),
                )
                last_timestamp = timestamp

                result = detector.detect_for_video(
                    mp_image,
                    timestamp,
                )

                face_visible = bool(result.face_landmarks)

                # ==================================
                # FACE DETECTED
                # ==================================

                if face_visible:

                    landmarks = result.face_landmarks[0]

                    if nose_cursor.enabled:
                        try:
                            nose_cursor.update(landmarks, frame)
                        except pyautogui.FailSafeException:
                            nose_cursor.stop()
                            disable_mouse()
                            status = "SAFETY STOP"
                    shapes = {}

                    if result.face_blendshapes:
                        shapes = {
                            item.category_name: item.score
                            for item in result.face_blendshapes[0]
                        }

                    jaw_open = shapes.get("jawOpen", 0.0)
                    brow_score = shapes.get("browInnerUp", 0.0)

                    score, mouth_box = (
                        tongue.calculate_score(
                            frame,
                            landmarks,
                        )
                    )

                    draw_landmarks(frame, landmarks)

                    if mouth_box is not None:
                        x1, y1, x2, y2 = mouth_box

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 255, 0),
                            2,
                        )

                    # --------------------------
                    # AUTOMATIC CALIBRATION
                    # --------------------------

                    if calibrating:

                        calibrator.update(
                            score,
                            True,
                            frame,
                        )

                        if calibrator.stage == "failed":
                            calibrating = False
                            status = "Tongue calibration failed - press C to retry"
                            irisvoice.speak("Tongue calibration failed.")

                        elif calibrator.finished:
                            calibrating = False
                            waiting_for_release = True

                            reset_gestures()

                            status = (
                                "Calibrated - enable mouse"
                            )

                            print(
                                "CALIBRATION COMPLETE",
                                flush=True,
                            )
                            irisvoice.speak("Calibration completed.")

                    # --------------------------
                    # FACIAL MOUSE CONTROLS
                    # --------------------------

                    elif (
                        mouse.enabled
                        and calibrator.finished
                    ):

                        tongue_visible = (
                            tongue.threshold is not None
                            and tongue.is_tongue_score(score)
                            and jaw_open > 0.12
                        )

                        brows_neutral = (
                            brow_score
                            < eyebrows.release_threshold
                        )

                        if waiting_for_release:

                            if (
                                not tongue_visible
                                and brows_neutral
                            ):
                                waiting_for_release = False
                                reset_gestures()

                                print(
                                    "GESTURES READY",
                                    flush=True,
                                )

                        else:

                            # EYEBROW HIGHLIGHTING
                            eyebrow_action = (
                                eyebrows.update(
                                    shapes,
                                    now,
                                )
                            )

                            if eyebrow_action == "start":
                                mouse.start_highlight()

                            elif eyebrow_action == "stop":
                                mouse.stop_highlight()

                            # TONGUE RIGHT CLICK
                            right_click = tongue.update(
                                score,
                                jaw_open,
                                now,
                            )
    
                            if right_click:
                                print(
                                    "TONGUE GESTURE CONFIRMED | "
                                    f"Eyebrow highlighting={eyebrows.highlighting} | "
                                    f"Dragging={mouse.dragging}",
                                    flush=True
                                )

                                if (
                                    not eyebrows.highlighting
                                    and not mouse.dragging
                                ):
                                    print("RIGHT CLICK!", flush=True)
                                    mouse.right_click()
                                else:
                                    print(
                                        "Right-click blocked by highlighting/dragging",
                                        flush=True
                                    )

                            # DOUBLE BLINK LEFT CLICK
                            if blink_detector.enabled:

                                should_click, _ = (
                                    blink_detector.update(
                                        landmarks
                                    )
                                )

                                if (
                                    should_click
                                    and not mouse.dragging
                                    and not eyebrows.highlighting
                                ):
                                    print(
                                        "DOUBLE BLINK DETECTED",
                                        flush=True,
                                    )

                                    pyautogui.click(
                                        button="left"
                                    )

                    else:
                        reset_gestures()

                    # --------------------------
                    # VISUAL FEEDBACK
                    # --------------------------

                    draw_text(
                        frame,
                        f"Tongue: {score:.2f}",
                        40,
                    )

                    draw_text(
                        frame,
                        f"Eyebrows: {brow_score:.2f}",
                        70,
                    )

                    if mouse.dragging:
                        draw_text(
                            frame,
                            "HIGHLIGHTING",
                            100,
                            (0, 255, 0),
                        )

                    if tongue.detected:
                        draw_text(
                            frame,
                            "TONGUE DETECTED",
                            130,
                            (0, 255, 255),
                        )

                # ==================================
                # FACE LOST
                # ==================================

                else:
                    nose_cursor.face_lost()
                    reset_gestures()
                    waiting_for_release = True
                    
                    if calibrating:
                        calibrator.update(
                            None,
                            False,
                            frame,
                        )

                    draw_text(
                        frame,
                        "No face detected",
                        40,
                        (0, 0, 255),
                    )
                    
                    
                # ===========
                # face tilt
                # ===========

                # ==================================
                # STATUS DISPLAY
                # ==================================

                height = frame.shape[0]

                mouse_label = (
                    "ON" if mouse.enabled else "OFF"
                )
                blink_label = (
                    "ON" if blink_detector.enabled else "OFF"
                )
                typing_label = (
                    "ON" if speech_typer.enabled else "OFF"
                )

                if voice.ready:
                    voice_label = "READY"
                elif voice.error:
                    voice_label = "ERROR"
                else:
                    voice_label = "STARTING"

                draw_text(
                    frame,
                    f"Mouse: {mouse_label} | Blink: {blink_label}",
                    height - 105,
                )

                draw_text(
                    frame,
                    f"Voice: {voice_label} | Typing: {typing_label}",
                    height - 80,
                )

                draw_text(
                    frame,
                    status[:55],
                    height - 55,
                    (0, 255, 255),
                )

                draw_text(
                    frame,
                    "C Calibrate M Mouse X Off K Blink Q Quit",
                    height - 20,
                )

                # Voice-controlled command reference overlay.
                draw_command_menu(frame, menu_state["current"])

                cv2.imshow(
                    "AI Assisted Computer Controller",
                    frame,
                )

                # ==================================
                # KEYBOARD CONTROLS
                # ==================================

                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):
                    begin_calibration()

                elif key == ord("m"):
                    enable_mouse()

                elif key == ord("x"):
                    disable_mouse()

                elif key == ord("k"):
                    set_blink(
                        not blink_detector.enabled
                    )
                    
                elif key == ord("n"):
                    if nose_cursor.enabled:
                        nose_cursor.stop()
                    else:
                        nose_cursor.start()

                elif key == ord("r"):
                    nose_cursor.recalibrate()

                elif key == ord("f"):
                    nose_cursor.freeze()

                elif key == ord("u"):
                    nose_cursor.unfreeze()

                elif key == ord("q"):
                    break

    # ==========================================
    # ERROR HANDLING AND CLEANUP
    # ==========================================

    except pyautogui.FailSafeException:
        print(
            "SAFETY STOP: PyAutoGUI failsafe triggered",
            flush=True,
        )

    except KeyboardInterrupt:
        print("Stopped by user", flush=True)

    finally:
        speech_typer.stop()
        nose_cursor.stop()
        mouse.cleanup()
        voice.stop()

        camera.release()
        cv2.destroyAllWindows()

        print(
            "AI Computer Controller stopped",
            flush=True,
        )


if __name__ == "__main__":
    run_detector()