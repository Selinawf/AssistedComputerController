
import pyautogui


class SpeechTyper:

    def __init__(self):
        self.enabled = False

        # Keep the emergency corner failsafe
        pyautogui.FAILSAFE = True

    def start(self):
        self.enabled = True

        print(
            "SPEECH TYPING: ON",
            flush=True
        )

    def stop(self):
        self.enabled = False

        print(
            "SPEECH TYPING: OFF",
            flush=True
        )

    def type_text(self, text):
        if not self.enabled:
            return

        text = text.strip()

        if not text:
            return

        print(
            f"TYPING: {text}",
            flush=True
        )

        # Type into the currently focused
        # text field, not the webcam window.
        pyautogui.write(
            text + " ",
            interval=0.001
        )

    def new_line(self):
        if not self.enabled:
            return

        pyautogui.press("enter")


    def delete_word(self, count=1):
        if not self.enabled:
            return

        count = max(1, int(count))

        # macOS: delete the previous word repeatedly
        for _ in range(count):
            pyautogui.hotkey("option", "backspace")

        print(f"DELETED LAST {count} WORD(S)", flush=True)

