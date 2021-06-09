import cv2
import mediapipe as mp # media pipe library to detect the feature points on the face
import time

cap = cv2.VideoCapture(0) # it captures through inbuilt webcam


drawFace = mp.solutions.drawing_utils
drawFaceMesh = mp.solutions.face_mesh
faceMesh = drawFaceMesh.FaceMesh(False, 5, 0.5, 0.5) # maximum 5 faces will be detected.
# faceMesh = drawFaceMesh.FaceMesh(max_num_faces=2)

pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultMesh = faceMesh.process(imgRGB)
    if resultMesh.multi_face_landmarks:
        for face in resultMesh.multi_face_landmarks:
            drawFace.draw_landmarks(img, face, drawFaceMesh.FACE_CONNECTIONS)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    frame_count = int(cv2.CAP_PROP_FRAME_COUNT)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}',(20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
    cv2.imshow("Figure1", img)
    cv2.waitKey(27)

