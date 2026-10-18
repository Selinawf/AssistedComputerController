
"""Reference-based tongue gesture measurement."""

import cv2
import numpy as np


class TongueDetector:
    ROI_SIZE = (80, 100)

    def __init__(self):
        self.baseline = None
        self.threshold = None
        self.direction = 1

        self.reference = None
        self.last_roi = None
        self.score = 0.0
        self.detected = False

        self.hold_seconds = 0.6
        self.cooldown_seconds = 1.0

        self.gesture_start = None
        self.last_click = float("-inf")
        self.armed = True

    def clear_reference(self):
        self.reference = None
        self.last_roi = None
        self.baseline = None
        self.threshold = None
        self.score = 0.0
        self.reset()

    def set_reference(self, frames):
        """Learn the appearance of the chin with tongue hidden."""

        if len(frames) < 10:
            return False

        reference = np.median(
            np.stack(frames),
            axis=0
        ).astype(np.uint8)

        self.reference = cv2.GaussianBlur(
            reference,
            (5, 5),
            0
        )
        return True

    def calculate_score(self, frame, landmarks):
        """
        Returns (score, mouth_box).

        Score = percentage of pixels changed
        from the tongue-hidden reference image.
        """

        height, width = frame.shape[:2]

        if len(landmarks) <= 291:
            self.last_roi = None
            self.score = 0.0
            return self.score, None

        left = landmarks[61]
        right = landmarks[291]
        upper = landmarks[13]
        lower = landmarks[14]

        mouth_left = min(left.x, right.x) * width
        mouth_right = max(left.x, right.x) * width
        mouth_width = mouth_right - mouth_left

        if mouth_width < 8:
            self.last_roi = None
            self.score = 0.0
            return self.score, None

        center_x = (mouth_left + mouth_right) / 2
        upper_y = upper.y * height
        lower_y = lower.y * height

        opening = max(
            1.0,
            abs(lower_y - upper_y)
        )

        # Keep the rectangle that covers your tongue.
        x1 = max(
            0,
            int(center_x - mouth_width * 0.30)
        )
        x2 = min(
            width,
            int(center_x + mouth_width * 0.30)
        )
        y1 = max(
            0,
            int(upper_y + opening * 0.15)
        )
        y2 = min(
            height,
            int(lower_y + mouth_width * 0.65)
        )

        box = (x1, y1, x2, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            self.last_roi = None
            self.score = 0.0
            return self.score, None

        roi = frame[y1:y2, x1:x2]

        # cv2.imshow(
        #     "Tongue Detection Region",
        #     roi
        # )

        # Resize every sample to the same dimensions.
        resized = cv2.resize(
            roi,
            self.ROI_SIZE,
            interpolation=cv2.INTER_AREA
        )

        self.last_roi = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2LAB
        )

        # No reference yet: calibration needs to
        # collect tongue-hidden frames first.
        if self.reference is None:
            self.score = 0.0
            return self.score, box

        current = cv2.GaussianBlur(
            self.last_roi,
            (5, 5),
            0
        )

        difference = cv2.absdiff(
            current,
            self.reference
        )

        # Measure lightness and colour changes.
        change = (
            0.4 * difference[:, :, 0].astype(np.float32)
            + 0.6 * difference[:, :, 1].astype(np.float32)
            + 0.6 * difference[:, :, 2].astype(np.float32)
        )

        mask = (
            (change > 24).astype(np.uint8)
            * 255
        )

        # Remove tiny isolated changed pixels.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        self.score = (
            100.0
            * cv2.countNonZero(mask)
            / mask.size
        )

        # White = changed from reference.
        # Black = similar to reference.
        # cv2.imshow(
        #     "Tongue Change Mask",
        #     mask
        # )

        return float(self.score), box

    def is_tongue_score(self, score):
        return (
            self.threshold is not None
            and score > self.threshold
        )

    def update(self, score, jaw_open, now):
        """Return True once for a sustained tongue gesture."""

        if self.threshold is None:
            self.reset()
            return False

        self.detected = (
            self.is_tongue_score(score)
            and jaw_open > 0.12
        )

        # Show what blocks the tongue gesture
        if not self.detected:
            self.reset()
            return False

        if not self.armed:
            return False

        if self.gesture_start is None:
            self.gesture_start = now

        if (
            now - self.gesture_start >= self.hold_seconds
            and now - self.last_click >= self.cooldown_seconds
        ):
            self.armed = False
            self.last_click = now
            return True

        return False

    def reset(self):
        """Reset gesture timing without losing calibration."""

        self.gesture_start = None
        self.armed = True
        self.detected = False