
class MouseActions:

    def __init__(self, enabled=False):
        self.enabled = False
        self.dragging = False
        self.pyautogui = None

        if enabled:
            self.set_enabled(True)

    # ======================================
    # ENABLE / DISABLE MOUSE
    # ======================================

    def set_enabled(self, enabled):

        if not enabled:
            self.stop_highlight()
            self.enabled = False

            print("MOUSE CONTROLS: OFF")
            return

        if self.pyautogui is None:
            import pyautogui

            self.pyautogui = pyautogui
            self.pyautogui.FAILSAFE = True

        self.enabled = True

        print("MOUSE CONTROLS: ON")

    # ======================================
    # RIGHT CLICK
    # ======================================

    def right_click(self):

        if self.dragging:
            return

        if self.enabled:
            print("RIGHT CLICK DETECTED!")

            assert self.pyautogui is not None

            self.pyautogui.click(
                button="right"
            )

    # ======================================
    # START HIGHLIGHT
    # ======================================

    def start_highlight(self):

        if self.dragging:
            return

        if self.enabled:
            print("HIGHLIGHT STARTED!")

            assert self.pyautogui is not None

            self.pyautogui.mouseDown(
                button="left"
            )

            self.dragging = True

    # ======================================
    # STOP HIGHLIGHT
    # ======================================

    def stop_highlight(self):

        if not self.dragging:
            return

        try:
            if self.pyautogui is not None:
                self.pyautogui.mouseUp(
                    button="left"
                )

        finally:
            self.dragging = False

            print("HIGHLIGHT STOPPED!")

    # ======================================
    # CLEANUP
    # ======================================

    def cleanup(self):
        self.stop_highlight()
        self.enabled = False