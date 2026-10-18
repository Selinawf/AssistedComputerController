
import math
import time


class BlinkDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        # Initial thresholds - calibrate using live EAR readings
        self.close_threshold = 0.24
        self.open_threshold = 0.27

        # Blink timing
        self.min_blink_duration = 0.05
        self.max_blink_duration = 0.45
        self.double_blink_window = 1.00
        self.min_blink_gap = 0.12
        self.click_cooldown = 0.65

        self.enabled = True
        self.last_click_time = float("-inf")

        # Debugging information
        self.current_ear = 0.0
        self.left_ear = 0.0
        self.right_ear = 0.0
        self.blink_count = 0

        self.reset()

    @staticmethod
    def distance(a, b):
        return math.hypot(
            a.x - b.x,
            a.y - b.y
        )

    def eye_aspect_ratio(self, landmarks, indices):
        p = [landmarks[i] for i in indices]

        vertical1 = self.distance(p[1], p[5])
        vertical2 = self.distance(p[2], p[4])
        horizontal = self.distance(p[0], p[3])

        if horizontal <= 1e-6:
            return None

        return (
            (vertical1 + vertical2)
            / (2.0 * horizontal)
        )

    def reset(self):
        self.eyes_closed = False
        self.blink_start = None
        self.first_blink_time = None
        self.last_blink_time = None
        self.blink_count = 0

    def toggle(self):
        self.enabled = not self.enabled
        self.reset()

        print(
            "Blink clicking ENABLED"
            if self.enabled
            else "Blink clicking DISABLED",
            flush=True
        )

    def update(self, landmarks):
        """
        Returns:
            should_click: bool
            freeze_cursor: bool
        """

        if not self.enabled:
            return False, False

        if landmarks is None or len(landmarks) <= 387:
            self.reset()
            return False, True

        now = time.monotonic()

        left = self.eye_aspect_ratio(
            landmarks, self.LEFT_EYE
        )
        right = self.eye_aspect_ratio(
            landmarks, self.RIGHT_EYE
        )

        if left is None or right is None:
            self.reset()
            return False, True

        self.left_ear = left
        self.right_ear = right

        ear = (left + right) / 2.0
        self.current_ear = ear

        should_click = False

        # Expire unfinished double-blink sequence
        if self.first_blink_time is not None:
            elapsed = now - self.first_blink_time

            if elapsed > self.double_blink_window:
                self.first_blink_time = None
                self.blink_count = 0

        # Eyes close
        if (
            not self.eyes_closed
            and ear < self.close_threshold
        ):
            self.eyes_closed = True
            self.blink_start = now

        # Eyes reopen
        elif (
            self.eyes_closed
            and ear > self.open_threshold
        ):
            self.eyes_closed = False

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

                # Reject impossible rapid repeat blinks
                if (
                    self.last_blink_time is not None
                    and now - self.last_blink_time
                    < self.min_blink_gap
                ):
                    freeze = self.first_blink_time is not None
                    return False, freeze

                self.last_blink_time = now

                if self.first_blink_time is None:
                    # First blink
                    self.first_blink_time = now
                    self.blink_count = 1

                    print(
                        f"BLINK 1 | EAR: {ear:.3f}"
                        f" | Duration: {duration:.3f}s",
                        flush=True
                    )

                else:
                    # Second blink
                    interval = now - self.first_blink_time

                    if interval <= self.double_blink_window:
                        if (
                            now - self.last_click_time
                            >= self.click_cooldown
                        ):
                            should_click = True
                            self.last_click_time = now

                            print(
                                "DOUBLE BLINK CONFIRMED!",
                                flush=True
                            )

                    self.first_blink_time = None
                    self.blink_count = 0

            else:
                print(
                    f"Blink rejected: {duration:.3f}s",
                    flush=True
                )

                self.first_blink_time = None
                self.blink_count = 0

        # Prevent cursor movement during blinking
        freeze_cursor = (
            self.eyes_closed
            or self.first_blink_time is not None
            or should_click
        )

        return should_click, freeze_cursor