
import queue
import threading

import pyttsx3


class VoiceAssistant:
    def __init__(self, rate=175, volume=1.0, voice_index=0):
        self.messages = queue.Queue()
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self.thread = threading.Thread(
            target=self._speak_worker,
            daemon=True,
        )
        self.thread.start()

    def speak(self, text):
        if text:
            self.messages.put(text)

    def _speak_worker(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)

        
        voices = engine.getProperty("voices")

        if isinstance(voices, list) and voices:
            index = max(
                0,
                min(self.voice_index, len(voices) - 1),
            )

            selected_voice = voices[index]
            engine.setProperty("voice", selected_voice.id)


        while True:
            message = self.messages.get()

            try:
                if message is None:
                    break

                engine.say(message)
                engine.runAndWait()
            except Exception as error:
                print(f"Voice output error: {error}", flush=True)
            finally:
                self.messages.task_done()

        engine.stop()

    def stop(self):
        """Finish queued announcements and stop the speech thread."""
        self.messages.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=5)
