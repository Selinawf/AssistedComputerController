
class MouseActions:
    def __init__(self, enabled=False):
        self.enabled = False
        self.dragging = False
        self.pyautogui = None

        if enabled:
            self.set_enabled(True)

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

    def right_click(self):
        if self.dragging:
            return

        print("RIGHT CLICK DETECTED!")

        if self.enabled:
            assert self.pyautogui is not None
            self.pyautogui.click(button="right")

    def start_highlight(self):
        if self.dragging:
            return

        print("HIGHLIGHT STARTED!")

        if self.enabled:
            assert self.pyautogui is not None
            self.pyautogui.mouseDown(button="left")
            self.dragging = True

    def stop_highlight(self):
        if not self.dragging:
            return

        try:
            if self.pyautogui is not None:
                self.pyautogui.mouseUp(button="left")
        finally:
            self.dragging = False
            print("HIGHLIGHT STOPPED!")

    def cleanup(self):
        self.stop_highlight()
        self.enabled = False