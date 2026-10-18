"""Larger on-camera command reference menus for Iris."""

import cv2

VOICE_COMMANDS = [
    ("Calibration", "Computer start calibration"),
    ("Mouse", "Enable mouse / Disable mouse"),
    ("Blink", "Enable blink / Disable blink"),
    ("Nose", "Enable nose / Disable nose"),
    ("Nose position", "Recenter nose"),
    ("Cursor", "Freeze cursor / Unfreeze cursor"),
    ("Typing", "Start typing / Stop typing"),
    ("Dictation", "New line"),
    ("Scroll", "Scroll up / Scroll down"),
    ("Small scroll", "Scroll up/down a little"),
    ("Large scroll", "Scroll up/down a lot"),
    ("Speed", "Scroll faster / Scroll slower"),
    ("Stop", "Stop scrolling"),
    ("This menu", "Show voice commands"),
    ("Keyboard menu", "Show keyboard commands"),
    ("Close menu", "Hide commands"),
]

KEYBOARD_COMMANDS = [
    ("C", "Start calibration"),
    ("M", "Enable mouse"),
    ("X", "Disable mouse"),
    ("K", "Toggle blinking"),
    ("N", "Toggle nose tracking"),
    ("R", "Recenter nose tracking"),
    ("F", "Freeze nose cursor"),
    ("U", "Unfreeze nose cursor"),
    ("Q", "Quit Iris"),
    ("Voice", "Show voice commands"),
    ("Voice", "Show keyboard commands"),
    ("Voice", "Hide commands"),
]


def _fitting_scale(text, font, preferred_scale, max_width):
    """Keep long phrases fully visible rather than clipping them."""
    scale = preferred_scale
    while scale > 0.35:
        text_width = cv2.getTextSize(text, font, scale, 1)[0][0]
        if text_width <= max_width:
            break
        scale -= 0.025
    return max(0.35, scale)


def draw_command_menu(frame, menu_type):
    """Draw readable Iris commands over the existing webcam window."""
    if menu_type not in ("voice", "keyboard"):
        return frame

    height, width = frame.shape[:2]
    entries = VOICE_COMMANDS if menu_type == "voice" else KEYBOARD_COMMANDS
    font = cv2.FONT_HERSHEY_SIMPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (19, 28, 43), -1)
    cv2.addWeighted(overlay, 0.94, frame, 0.06, 0, frame)

    title = "IRIS - VOICE COMMANDS" if menu_type == "voice" else "IRIS - KEYBOARD COMMANDS"
    title_scale = _fitting_scale(title, font, 0.95, width - 30)
    cv2.putText(frame, title, (15, 37), font, title_scale, (100, 230, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (15, 48), (width - 15, 48), (80, 130, 160), 1)

    # Two columns, each command on two lines. Larger than the old 0.32-sized text.
    split = (len(entries) + 1) // 2
    columns = (entries[:split], entries[split:])
    top, bottom = 74, height - 38
    row_height = (bottom - top) / max(1, split)
    column_width = width // 2
    max_text_width = column_width - 30

    preferred = min(0.78, max(0.55, width / 1050))
    # Keep labels and phrases apart on shorter camera windows.
    preferred = min(preferred, max(0.45, row_height / 45))

    for column_index, column_entries in enumerate(columns):
        x = 15 + column_index * column_width
        for index, (label, phrase) in enumerate(column_entries):
            row_top = int(top + index * row_height)
            label_scale = _fitting_scale(label + ":", font, preferred, max_text_width)
            phrase_scale = _fitting_scale(phrase, font, preferred, max_text_width)
            cv2.putText(frame, label + ":", (x, row_top), font, label_scale,
                        (130, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, phrase, (x, row_top + max(18, int(row_height * 0.45))),
                        font, phrase_scale, (255, 255, 255), 1, cv2.LINE_AA)

    footer = 'Say "Hide commands" to close'
    footer_scale = _fitting_scale(footer, font, 0.62, width - 30)
    cv2.putText(frame, footer, (15, height - 10), font, footer_scale,
                (150, 255, 170), 1, cv2.LINE_AA)
    return frame
