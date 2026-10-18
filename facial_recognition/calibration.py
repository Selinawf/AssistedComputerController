
"""Tongue calibration using a tongue-hidden image reference."""

import statistics
import time
import cv2


class AutoCalibrator:
    def __init__(self, tongue_detector):
        self.tongue = tongue_detector

        self.ready_seconds = 3.0
        self.sample_seconds = 2.5
        self.max_attempts = 3

        self.reset()

    def reset(self):
        self.tongue.clear_reference()

        self.stage = "ready"
        self.stage_start = None
        self.samples = []
        self.reference_frames = []

        self.failed_attempts = 0
        self.finished = False

        self.message = (
            "Prepare: mouth OPEN, tongue HIDDEN"
        )

        print(
            "Calibration restarted!",
            flush=True
        )

    def begin_stage(self, stage, now):
        self.stage = stage
        self.stage_start = now
        self.samples = []

        if stage == "baseline":
            self.reference_frames = []
            self.tongue.clear_reference()

    def update(self, score, face_visible, frame):
        now = time.monotonic()

        if self.finished or self.stage == "failed":
            self.draw(frame)
            return self.finished

        if (
            not face_visible
            or score is None
            or self.tongue.last_roi is None
        ):
            self.begin_stage("ready", now)
            self.tongue.clear_reference()

            self.message = (
                "Face lost. Look at camera; tongue hidden."
            )

            self.draw(frame)
            return False

        if self.stage_start is None:
            self.stage_start = now

        elapsed = now - self.stage_start

        # -------------------------------------
        # STAGE 1: GET READY
        # -------------------------------------

        if self.stage == "ready":
            self.message = (
                "OPEN mouth, tongue HIDDEN. Hold still."
            )

            if elapsed >= self.ready_seconds:
                self.begin_stage("baseline", now)

        # -------------------------------------
        # STAGE 2: LEARN TONGUE-HIDDEN IMAGE
        # -------------------------------------

        elif self.stage == "baseline":
            self.message = (
                "Recording chin with tongue HIDDEN"
            )

            self.reference_frames.append(
                self.tongue.last_roi.copy()
            )

            if elapsed >= self.sample_seconds:
                if not self.tongue.set_reference(
                    self.reference_frames
                ):
                    self.failure(
                        "Not enough reference frames",
                        now
                    )
                else:
                    self.begin_stage(
                        "check_baseline",
                        now
                    )

        # -------------------------------------
        # STAGE 3: MEASURE NORMAL MOVEMENT
        # -------------------------------------

        elif self.stage == "check_baseline":
            self.message = (
                "Still tongue HIDDEN: checking movement"
            )

            self.samples.append(float(score))

            if elapsed >= self.sample_seconds:
                self.baseline = statistics.median(
                    self.samples
                )

                self.baseline_noise = statistics.median(
                    abs(x - self.baseline)
                    for x in self.samples
                )

                print(
                    f"Baseline change: {self.baseline:.2f}% "
                    f"(noise {self.baseline_noise:.2f}%)",
                    flush=True
                )

                self.begin_stage(
                    "prepare_tongue",
                    now
                )

        # -------------------------------------
        # STAGE 4: PREPARE TONGUE
        # -------------------------------------

        elif self.stage == "prepare_tongue":
            remaining = max(
                0.0,
                self.ready_seconds - elapsed
            )

            self.message = (
                f"Stick tongue OUT in {remaining:.1f}s"
            )

            if elapsed >= self.ready_seconds:
                self.begin_stage(
                    "tongue",
                    now
                )

        # -------------------------------------
        # STAGE 5: MEASURE TONGUE EXTENSION
        # -------------------------------------

        elif self.stage == "tongue":
            self.message = (
                "Hold tongue OUT toward chin"
            )

            self.samples.append(float(score))

            if elapsed >= self.sample_seconds:
                tongue_value = statistics.median(
                    self.samples
                )

                tongue_noise = statistics.median(
                    abs(x - tongue_value)
                    for x in self.samples
                )

                required = max(
                    6.0,
                    3.0 * (
                        self.baseline_noise + tongue_noise
                    )
                )

                difference = (
                    tongue_value - self.baseline
                )

                #print(
                    #f"Tongue change: {tongue_value:.2f}% "
                    #f"| difference {difference:+.2f}% "
                   # f"| minimum {required:.2f}%",
                   # flush=True
              #  )

                if difference >= required:
                    self.tongue.baseline = self.baseline

                    self.tongue.threshold = (
                        self.baseline + tongue_value
                    ) / 2.0

                    self.tongue.direction = 1
                    self.tongue.reset()

                    self.finished = True
                    self.stage = "done"

                    self.message = (
                        "Calibrated! Enable mouse with M."
                    )

                    print(
                        "Calibration successful!",
                        flush=True
                    )

                else:
                    self.failure(
                        "Tongue and normal chin overlap",
                        now
                    )

        self.draw(frame)
        return self.finished

    def failure(self, reason, now):
        self.failed_attempts += 1

        self.tongue.clear_reference()

        print(
            f"Calibration failed "
            f"({self.failed_attempts}/{self.max_attempts}): "
            f"{reason}",
            flush=True
        )

        if self.failed_attempts >= self.max_attempts:
            self.stage = "failed"
            self.message = (
                "Calibration failed - press C to retry"
            )
        else:
            self.begin_stage("ready", now)
            self.message = (
                "Retry: mouth OPEN, tongue HIDDEN"
            )

    def draw(self, frame):
        height, width = frame.shape[:2]

        cv2.rectangle(
            frame,
            (10, height - 160),
            (width - 10, height - 125),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            self.message[:65],
            (20, height - 136),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )