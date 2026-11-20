<<<<<<< HEAD

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Webcam tidak dapat dibuka")

print("Webcam berhasil dibuka!")
print("Tekan Q untuk keluar")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame")
            break

        cv2.imshow("Test Webcam BISINDO", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
=======

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Webcam tidak dapat dibuka")

print("Webcam berhasil dibuka!")
print("Tekan Q untuk keluar")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame")
            break

        cv2.imshow("Test Webcam BISINDO", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
>>>>>>> 325f27dbe444994b8c42a10340ceb23e9118c9b8
