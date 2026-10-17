
import cv2
import time

print("STEP 1: Python started", flush=True)

print("STEP 2: Opening webcam...", flush=True)
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

print("STEP 3: Camera opened:", camera.isOpened(), flush=True)

if not camera.isOpened():
    print("ERROR: Cannot open camera", flush=True)
    raise SystemExit(1)

print("STEP 4: Waiting for camera...", flush=True)
time.sleep(2)

print("STEP 5: Reading first frame...", flush=True)
success, frame = camera.read()

print("STEP 6: Frame captured:", success, flush=True)

if success and frame is not None:
    print("SUCCESS: Displaying webcam", flush=True)

    while True:
        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        success, frame = camera.read()

        if not success:
            print("ERROR: Lost camera frames", flush=True)
            break
else:
    print("ERROR: Camera opened but no video received", flush=True)

camera.release()
cv2.destroyAllWindows()
print("Program ended", flush=True)
