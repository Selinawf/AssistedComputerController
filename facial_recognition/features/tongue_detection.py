
import cv2


class TongueDetector:
    def __init__(self):
        self.baseline = None
        self.threshold = None
        self.score = 0.0

        self.hold_seconds = 0.6
        self.cooldown_seconds = 1.0

        self.gesture_start = None
        self.last_click = float("-inf")
        self.armed = True
        self.detected = False

    def calculate_score(self, frame, landmarks):
        height, width = frame.shape[:2]

        left = landmarks[61]
        right = landmarks[291]
        upper = landmarks[13]
        lower = landmarks[14]

        x1 = int(min(left.x, right.x) * width)
        x2 = int(max(left.x, right.x) * width)

        mouth_width = max(1, x2 - x1)

        y1 = int(upper.y * height)
        y2 = int(lower.y * height + mouth_width * 0.45)

        x1 = max(0, x1)
        x2 = min(width, x2)
        y1 = max(0, y1)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            self.score = 0.0
            return 0.0, None

        roi = frame[y1:y2, x1:x2]

        # Convert to LAB colour space
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        # A channel measures green-to-red colour
        a_channel = lab[:, :, 1]

        # Use the centre of the mouth region
        roi_height, roi_width = a_channel.shape

        center = a_channel[
            roi_height // 4: 3 * roi_height // 4,
            roi_width // 4: 3 * roi_width // 4
        ]

        if center.size == 0:
            self.score = 0.0
            return 0.0, None

        # Average redness value
        self.score = float(center.mean())

        return self.score, (x1, y1, x2, y2)

    def calibrate_baseline(self):
        self.baseline = self.score
        self.threshold = None
        self.reset()

        print(f"Baseline: {self.baseline:.3f}")

    def calibrate_tongue(self):
        if self.baseline is None:
            print("Press B first to record baseline.")
            return

        tongue_sample = self.score

        print(f"Tongue sample: {tongue_sample:.3f}")

        if tongue_sample > self.baseline + 2.0:
            self.threshold = (
                self.baseline + tongue_sample
            ) / 2

            self.reset()

            print("Calibration successful!")
            print(f"Threshold: {self.threshold:.3f}")
        else:
            self.threshold = None
            print("Calibration failed. Try again.")

    def update(self, score, jaw_open, now):
        if self.threshold is None:
            self.reset()
            return False

        self.detected = (
            score > self.threshold
            and jaw_open > 0.12
        )

        if not self.detected:
            self.reset()
            return False

        if not self.armed:
            return False

        if self.gesture_start is None:
            self.gesture_start = now

        held = now - self.gesture_start
        cooldown = now - self.last_click

        if (
            held >= self.hold_seconds
            and cooldown >= self.cooldown_seconds
        ):
            self.armed = False
            self.last_click = now
            return True

        return False

    def reset(self):
        self.gesture_start = None
        self.armed = True
        self.detected = False