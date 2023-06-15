import cv2
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

p_time = 0


def place_fps():
    global p_time

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_lms)

            for id, lm in enumerate(face_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                print(id, cx, cy)

        place_fps()

        cv2.imshow("Image", img)
        cv2.waitKey(1)
