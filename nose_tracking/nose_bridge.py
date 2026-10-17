
import math
import cv2
import pyautogui

# Same settings as your original nose tracker
DEAD_ZONE = 0.025
SMOOTHING = 0.25
NOSE_TIP = 1
RANGE_X = 0.15
RANGE_Y = 0.12

pyautogui.FAILSAFE = True


class NoseCursor:
    def __init__(self):
        self.enabled = False
        self.frozen = False
        self.calibration_x = None
        self.calibration_y = None

        self.screen_w, self.screen_h = pyautogui.size()
        self.current_x, self.current_y = map(
            float, pyautogui.position()
        )

    def start(self):
        self.enabled = True
        self.frozen = False
        self.recalibrate()
        print("Nose tracking ON", flush=True)

    def stop(self):
        self.enabled = False
        self.frozen = False
        print("Nose tracking OFF", flush=True)

    def recalibrate(self):
        self.calibration_x = None
        self.calibration_y = None
        print("Nose calibration reset", flush=True)

    def freeze(self):
        self.frozen = True
        print("Cursor frozen", flush=True)

    def unfreeze(self):
        self.frozen = False
        self.recalibrate()
        print("Cursor unfrozen", flush=True)

    def face_lost(self):
    # Reset only once when the face is lost
        if (
            self.calibration_x is not None
            or self.calibration_y is not None
        ):
            self.calibration_x = None
            self.calibration_y = None

    def update(self, landmarks, frame):
        if not self.enabled or self.frozen:
            return

        h, w = frame.shape[:2]

        nose = landmarks[NOSE_TIP]
        nx, ny = nose.x, nose.y

        if not (
            math.isfinite(nx)
            and math.isfinite(ny)
        ):
            return

        # First frame establishes neutral position
        if self.calibration_x is None:
            self.calibration_x = nx
            self.calibration_y = ny
            self.current_x, self.current_y = map(
                float, pyautogui.position()
            )
            return

        dx = nx - self.calibration_x
        dy = ny - self.calibration_y

        # Your original dead-zone calculation
        if abs(dx) < DEAD_ZONE:
            dx = 0.0
        else:
            dx -= math.copysign(DEAD_ZONE, dx)

        if abs(dy) < DEAD_ZONE:
            dy = 0.0
        else:
            dy -= math.copysign(DEAD_ZONE, dy)

        # Your original screen mapping
        target_x = (
            self.screen_w / 2
            + (dx / RANGE_X) * (self.screen_w / 2)
        )

        target_y = (
            self.screen_h / 2
            + (dy / RANGE_Y) * (self.screen_h / 2)
        )

        margin = 10

        target_x = max(
            margin,
            min(self.screen_w - margin, target_x)
        )
        target_y = max(
            margin,
            min(self.screen_h - margin, target_y)
        )

        self.current_x += (
            target_x - self.current_x
        ) * SMOOTHING

        self.current_y += (
            target_y - self.current_y
        ) * SMOOTHING

        pyautogui.moveTo(
            int(self.current_x),
            int(self.current_y)
        )

        # Draw the nose indicator
        cv2.circle(
            frame,
            (int(nx * w), int(ny * h)),
            8,
            (0, 255, 0),
            -1
        )
