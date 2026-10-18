import math
import time


class BlinkDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        # Adjust these if necessary
        self.close_threshold = 0.24       # Detect slightly less complete eye closures
        self.open_threshold = 0.27        # Recognize reopening
        self.min_blink_duration = 0.03    # Accept quicker blinks
        self.max_blink_duration = 0.50
        self.double_blink_window = 0.80   # More time for second blink

        # Blink clicking starts enabled
        self.enabled = True

        self.eyes_closed = False
        self.blink_start = None
        self.first_blink_time = None
        self.last_click_time = float("-inf")       
        self.click_cooldown = 0.8

    def distance(self, a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def eye_aspect_ratio(self, landmarks, indices):
        p = [landmarks[i] for i in indices]

        vertical1 = self.distance(p[1], p[5])
        vertical2 = self.distance(p[2], p[4])
        horizontal = self.distance(p[0], p[3])

        if horizontal <= 1e-6:
            return 0

        return (vertical1 + vertical2) / (2 * horizontal)

    def reset(self):
        self.eyes_closed = False
        self.blink_start = None
        self.first_blink_time = None

    def toggle(self):
        self.enabled = not self.enabled
        self.reset()

        print(
            "Blink clicking ENABLED"
            if self.enabled
            else "Blink clicking DISABLED"
        )

    def update(self, landmarks):
        """
        Returns:
            should_click: bool
            freeze_cursor: bool
        """

        if not self.enabled:
            return False, False

        if landmarks is None:
            self.reset()
            return False, True

        now = time.monotonic()

        left_ear = self.eye_aspect_ratio(
            landmarks, self.LEFT_EYE
        )
        right_ear = self.eye_aspect_ratio(
            landmarks, self.RIGHT_EYE
        )

        ear = (left_ear + right_ear) / 2
        should_click = False

        # Expire an unfinished double blink
        if (
            self.first_blink_time is not None
            and now - self.first_blink_time
            > self.double_blink_window
        ):
            self.first_blink_time = None

        # Detect eyes closing
        if not self.eyes_closed and ear < self.close_threshold:
            self.eyes_closed = True
            self.blink_start = now

        # Detect eyes reopening
        elif self.eyes_closed and ear > self.open_threshold:
            self.eyes_closed = False

            # Make sure a valid blink start time exists.
            if self.blink_start is None:
                self.reset()
                return False, False

            duration = now - self.blink_start
            self.blink_start = None


            valid_blink = (
                self.min_blink_duration
                <= duration
                <= self.max_blink_duration
            )

            if valid_blink:
                if self.first_blink_time is None:
                    self.first_blink_time = now

                elif (
                    now - self.first_blink_time
                    <= self.double_blink_window
                ):
                    if (
                        now - self.last_click_time
                        >= self.click_cooldown
                    ):
                        should_click = True
                        self.last_click_time = now

                    self.first_blink_time = None

            else:
                self.first_blink_time = None

        # Freeze while eyes are closed or
        # waiting for a possible second blink
        freeze_cursor = (
            self.eyes_closed
            or self.first_blink_time is not None
            or should_click
        )

        return should_click, freeze_cursor