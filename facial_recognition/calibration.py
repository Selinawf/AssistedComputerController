import time
import statistics
import cv2


class AutoCalibrator:
    def __init__(self, tongue_detector):
        self.tongue = tongue_detector
        self.stage = "ready"
        self.stage_start = None
        self.samples = []
        self.baseline = None
        self.finished = False

        self.ready_seconds = 3.0
        self.sample_seconds = 3.0

        self.message = "Preparing calibration..."

    def begin_stage(self, stage, now):
        self.stage = stage
        self.stage_start = now
        self.samples = []

    def update(self, score, face_visible, frame):
        now = time.monotonic()

        if self.finished:
            return True

        # Reset calibration if the face disappears
        if not face_visible or score is None:
            self.begin_stage("ready", now)
            self.baseline = None
            self.message = "Position your face in the camera"
            self.draw(frame)
            return False

        if self.stage_start is None:
            self.stage_start = now

        elapsed = now - self.stage_start

        if self.stage == "ready":
            remaining = max(0, self.ready_seconds - elapsed)

            self.message = f"Calibration starts in {remaining:.1f}s"

            if elapsed >= self.ready_seconds:
                self.begin_stage("baseline", now)

        elif self.stage == "baseline":
            self.message = "Open mouth, keep tongue hidden"
            self.samples.append(score)

            if elapsed >= self.sample_seconds:
                self.baseline = statistics.median(self.samples)

                print(f"Baseline: {self.baseline:.3f}")

                self.begin_stage("prepare_tongue", now)

        elif self.stage == "prepare_tongue":
            remaining = max(0, self.ready_seconds - elapsed)

            self.message = f"Stick tongue out in {remaining:.1f}s"

            if elapsed >= self.ready_seconds:
                self.begin_stage("tongue", now)

        elif self.stage == "tongue":
            self.message = "Hold your tongue OUT"
            self.samples.append(score)

            if elapsed >= self.sample_seconds:
                tongue_sample = statistics.median(self.samples)

                print(f"Tongue sample: {tongue_sample:.3f}")

                # Current detector assumes tongue score
                # increases compared with baseline
                if self.baseline is not None and tongue_sample > self.baseline + 2.0:
                    self.tongue.baseline = self.baseline
                    self.tongue.threshold = (self.baseline + tongue_sample) / 2
                    self.tongue.reset()

                    self.finished = True
                    self.message = "Calibration successful!"

                    print("Calibration successful!")
                    print(f"Threshold: " f"{self.tongue.threshold:.3f}")
                else:
                    self.message = "Retrying calibration"
                    print("Calibration failed. Retrying...")
                    self.begin_stage("ready", now)
                    self.baseline = None

        self.draw(frame)
        return self.finished

    
    def reset(self):
        self.stage = "ready"
        self.stage_start = None
        self.samples = []
        self.baseline = None
        self.finished = False

        self.tongue.baseline = None
        self.tongue.threshold = None
        self.tongue.reset()

        self.message = "Preparing calibration..."

        print("Calibration restarted!")

    def draw(self, frame):
        height, width = frame.shape[:2]

        cv2.rectangle(
            frame, (10, height - 115), (width - 10, height - 45), (30, 30, 30), -1
        )

        cv2.putText(
            frame,
            self.message,
            (20, height - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
