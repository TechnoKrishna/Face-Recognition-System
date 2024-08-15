import cv2
import os
import time

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('\n Enter user ID (e.g., 0, 1, 2, ...) and press <return> ==> ')
face_name = input(f'\n Enter name for user ID {face_id} and press <return> ==> ')

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

if not os.path.exists('dataset'):
    os.mkdir('dataset')

file_path = 'dataset/labels.txt'
file_exists = os.path.exists(file_path)
with open(file_path, 'a') as file:
    if file_exists:
        file.write('\n')
    file.write(face_name)

count = 0
frame_count = 0
start_time = time.time()

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        gray_face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray_face)
        count += 1

        frame_count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Img Captured: {count}", (img.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('image', img)

    k = cv2.waitKey(20) & 0xff
    if k == 27 or count >= 70:  # Press 'ESC' to stop
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
