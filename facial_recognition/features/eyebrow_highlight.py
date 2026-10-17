
class EyebrowHighlighter:
    def __init__(self):
        self.start_threshold = 0.55
        self.release_threshold = 0.30
        self.hold_seconds = 0.25

        self.highlighting = False
        self.raise_start = None
        self.score = 0.0

    def update(self, blendshapes, now):
        self.score = blendshapes.get(
            "browInnerUp", 0.0
        )

        if not self.highlighting:
            if self.score >= self.start_threshold:
                if self.raise_start is None:
                    self.raise_start = now

                if (
                    now - self.raise_start
                    >= self.hold_seconds
                ):
                    self.highlighting = True
                    self.raise_start = None
                    return "start"

            else:
                self.raise_start = None

        else:
            if self.score <= self.release_threshold:
                self.highlighting = False
                self.raise_start = None
                return "stop"

        return None

    def reset(self):
        was_highlighting = self.highlighting

        self.highlighting = False
        self.raise_start = None

        return "stop" if was_highlighting else None
