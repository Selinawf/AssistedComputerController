"""Offline Vosk voice listener in an isolated process (macOS friendly)."""

import json
import queue
import subprocess
import sys
import threading
import time
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "vosk-model-small-en-us-0.15"
TYPING_TIMEOUT = 10.0

COMMANDS = {
    "computer start calibration": "calibrate",
    "start calibration": "calibrate",
    "computer calibrate": "calibrate",
    "calibrate": "calibrate",
    "computer enable mouse": "enable_mouse",
    "enable mouse": "enable_mouse",
    "computer turn on mouse": "enable_mouse",
    "turn on mouse": "enable_mouse",
    "computer disable mouse": "disable_mouse",
    "disable mouse": "disable_mouse",
    "computer turn off mouse": "disable_mouse",
    "turn off mouse": "disable_mouse",
    "computer enable blink": "enable_blink",
    "enable blink": "enable_blink",
    "computer disable blink": "disable_blink",
    "disable blink": "disable_blink",
    # Nose commands — common Vosk misrecognitions
    "enable knows": "enable_nose",
    "enable know is": "enable_nose",
    "computer enable knows": "enable_nose",
    "computer enable know is": "enable_nose",
    "disable knows": "disable_nose",
    "disable know is": "disable_nose",
    "computer disable knows": "disable_nose",
    "computer disable know is": "disable_nose",
    "recenter knows": "recenter_nose",
    "recenter know is": "recenter_nose",
    "freeze knows": "freeze_cursor",
    "freeze know is": "freeze_cursor",
    "unfreeze knows": "unfreeze_cursor",
    "unfreeze know is": "unfreeze_cursor",
    # Scrolling
    "scroll up a little": "scroll_up_little",
    "scroll down a little": "scroll_down_little",
    "scroll up a lot": "scroll_up_lot",
    "scroll down a lot": "scroll_down_lot",
    "scroll up": "scroll_up",
    "scroll down": "scroll_down",
    "scroll faster": "scroll_faster",
    "scroll slower": "scroll_slower",
    "stop scrolling": "stop_scrolling",
    # Browser tab management
    "new tab": "new_tab",
    "open tab": "new_tab",
    "open a tab": "new_tab",
    "open new tab": "new_tab",
    "open a new tab": "new_tab",
    "create tab": "new_tab",
    "create a tab": "new_tab",
    "create a new tab": "new_tab",
    "computer new tab": "new_tab",
    "computer open new tab": "new_tab",
    "close tab": "close_tab",
    "close a tab": "close_tab",
    "close this tab": "close_tab",
    "close current tab": "close_tab",
    "close the current tab": "close_tab",
    "delete tab": "close_tab",
    "delete this tab": "close_tab",
    "remove tab": "close_tab",
    "computer close tab": "close_tab",
    "next tab": "next_tab",
    "switch to next tab": "next_tab",
    "go to next tab": "next_tab",
    "previous tab": "previous_tab",
    "switch to previous tab": "previous_tab",
    "go to previous tab": "previous_tab",

    # Reopen closed tab
    "reopen closed tab": "reopen_tab",
    "reopen last tab": "reopen_tab",
    "restore tab": "reopen_tab",
    "undo close tab": "reopen_tab",
    "computer reopen tab": "reopen_tab",
    "open last closed tab": "reopen_tab",

}
STOP_COMMANDS = {"computer stop typing", "stop typing", "stop dictation"}
NEW_LINE_COMMANDS = {"computer new line", "new line"}

def parse_delete_count(phrase):
    """Return a word count for delete commands, or None if not a match."""
    number_words = {
        "a": 1,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    match = re.fullmatch(
        r"(?:computer )?delete "
        r"(?:(?:last|previous) )?"
        r"(a|one|two|three|four|five|six|seven|eight|nine|ten|\d+) "
        r"words?",
        phrase,
    )

    if match:
        value = match.group(1)
        count = number_words.get(value)

        if count is None:
            count = int(value)

        # Safety limit to prevent accidental huge deletions.
        return max(1, min(count, 100))

    if phrase in DELETE_WORD_COMMANDS:
        return 1

    return None

def run_voice_worker():
    # Native sounddevice and Vosk libraries remain OUTSIDE the MediaPipe process.
    try:
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model, SetLogLevel
    except Exception as error:
        print(f"ERROR:Cannot import microphone/Vosk libraries: {error}", flush=True)
        return

    if not MODEL_PATH.is_dir():
        print(f"ERROR:Vosk model not found: {MODEL_PATH}", flush=True)
        return

    audio = queue.Queue(maxsize=60)

    def audio_callback(indata, frames, time_info, status):
        try:
            audio.put_nowait(bytes(indata))
        except queue.Full:
            pass

    try:
        SetLogLevel(-1)
        model = Model(str(MODEL_PATH))
        device = sd.query_devices(kind="input")
        sample_rate = int(device["default_samplerate"])
        recognizer = KaldiRecognizer(model, sample_rate)
        typing_mode = False
        last_typing_time = None

        def start_typing():
            nonlocal typing_mode, last_typing_time
            if not typing_mode:
                typing_mode = True
                print("CMD:start_typing", flush=True)
            last_typing_time = time.monotonic()

        def stop_typing():
            nonlocal typing_mode, last_typing_time
            if typing_mode:
                typing_mode = False
                last_typing_time = None
                print("CMD:stop_typing", flush=True)

        print(f"INFO:Input microphone: {device['name']}", flush=True)
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=4000,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            print("READY", flush=True)
            while True:
                if (
                    typing_mode
                    and last_typing_time is not None
                    and time.monotonic() - last_typing_time >= TYPING_TIMEOUT
                ):
                    print(
                        "INFO:Typing stopped after 10 seconds of inactivity", flush=True
                    )
                    stop_typing()

                try:
                    data = audio.get(timeout=0.2)
                except queue.Empty:
                    continue

                if not recognizer.AcceptWaveform(data):
                    continue

                phrase = json.loads(recognizer.Result()).get("text", "").strip().lower()
                if not phrase:
                    continue

                # Debugging: always show what Vosk actually recognized.
                print(f"HEARD:{phrase}", flush=True)

                if phrase in STOP_COMMANDS:
                    stop_typing()
                    continue
                
                delete_count = parse_delete_count(phrase)

                if delete_count is not None:
                    if typing_mode:
                        last_typing_time = time.monotonic()
                        print(f"CMD:delete_words:{delete_count}", flush=True)
                    continue

                # Tab commands must still work while dictation is active.
                # Require "computer" in typing mode to avoid confusing
                # ordinary dictated sentences with navigation commands.
                if typing_mode and phrase.startswith("computer "):
                    tab_command = COMMANDS.get(phrase.removeprefix("computer "))
                    if tab_command in {"new_tab", "close_tab", "next_tab", "previous_tab"}:
                        print(f"CMD:{tab_command}", flush=True)
                        last_typing_time = time.monotonic()
                        continue

                if typing_mode:
                    last_typing_time = time.monotonic()
                    if phrase in NEW_LINE_COMMANDS:
                        print("CMD:new_line", flush=True)
                    else:
                        print(f"TYPE:{phrase}", flush=True)
                    continue

                # Permit 'type', or 'type hello world'.
                if phrase == "type" or phrase.startswith("type "):
                    start_typing()
                    words = phrase[5:].strip() if phrase.startswith("type ") else ""
                    if words:
                        print(f"TYPE:{words}", flush=True)
                    continue

                # First attempt an exact match
                command = COMMANDS.get(phrase)

                if command is None:
                    words = phrase.split()

                    if "enable" in words and any(
                        word in words for word in ("nose", "knows")
                    ):
                        command = "enable_nose"

                    elif "disable" in words and any(
                        word in words for word in ("nose", "knows")
                    ):
                        command = "disable_nose"

                    elif "recenter" in words and any(
                        word in words for word in ("nose", "knows")
                    ):
                        command = "recenter_nose"

                    elif "freeze" in words and any(
                        word in words for word in ("nose", "knows")
                    ):
                        command = "freeze_cursor"

                    elif "unfreeze" in words and any(
                        word in words for word in ("nose", "knows")
                    ):
                        command = "unfreeze_cursor"

                # Accept common Vosk misrecognitions
                if command is None:
                    words = set(phrase.split())

                    if "calibration" in words and (
                        "computer" in words
                        or "computers" in words
                        or "start" in words
                        or "are" in words
                    ):
                        command = "calibrate"
                if command:
                    print(f"CMD:{command}", flush=True)
                else:
                    print("INFO:No matching command (see HEARD text above)", flush=True)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"ERROR:{type(error).__name__}: {error}", flush=True)


class VoiceListener:
    def __init__(self):
        self.process = None
        self.thread = None
        self.commands = queue.Queue()
        self.ready = False
        self.error = None
        self.typing_mode = False

    def start(self):
        if not MODEL_PATH.is_dir():
            self.error = f"Voice model missing: {MODEL_PATH}"
            print(self.error, flush=True)
            return False
        if self.process is not None and self.process.poll() is None:
            return True

        self.ready = False
        self.error = None
        self.typing_mode = False
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", "voice_control.listener", "--worker"],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.thread = threading.Thread(target=self._read_output, daemon=True)
            self.thread.start()
            print("Starting voice process...", flush=True)
            return True
        except Exception as error:
            self.error = str(error)
            print("Voice process error:", error, flush=True)
            return False

    def _read_output(self):
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                line = line.strip()
                if line == "READY":
                    self.ready = True
                    print("Voice controls ready!", flush=True)
                elif line.startswith("CMD:"):
                    command = line[4:]

                    if command.startswith("delete_words:"):
                        try:
                            count = int(command.split(":", 1)[1])
                            self.commands.put(("delete_words", count))
                        except ValueError:
                            print(f"Invalid delete command: {command}", flush=True)
                            continue
                    else:
                        self.commands.put(command)

                        if command == "start_typing":
                            self.typing_mode = True
                        elif command == "stop_typing":
                            self.typing_mode = False

                    print(f"Voice command: {command}", flush=True)
                elif line.startswith("TYPE:"):
                    words = line[5:]
                    if words:
                        self.commands.put(("type_text", words))
                elif line.startswith("HEARD:"):
                    print("[Voice heard]", line[6:], flush=True)
                elif line.startswith("ERROR:"):
                    self.error = line[6:]
                    print("Voice error:", self.error, flush=True)
                elif line.startswith("INFO:"):
                    print("[Voice]", line[5:], flush=True)
                elif line:
                    print("[Voice]", line, flush=True)
        except Exception as error:
            self.error = str(error)
        finally:
            self.ready = False
            self.typing_mode = False
            if (
                process.poll() is not None
                and process.returncode not in (0, -15)
                and self.error is None
            ):
                self.error = "Voice process stopped unexpectedly"

    def get_commands(self):
        pending = []
        while True:
            try:
                pending.append(self.commands.get_nowait())
            except queue.Empty:
                break
        return pending

    def stop(self):
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)
        self.process = None
        self.thread = None
        self.ready = False
        self.typing_mode = False


if __name__ == "__main__" and "--worker" in sys.argv:
    run_voice_worker()
