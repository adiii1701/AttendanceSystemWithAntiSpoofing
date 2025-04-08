import os
import pickle
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO

# === Firebase Setup ===
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-2e6b8-default-rtdb.firebaseio.com/"
})

# === Load Background and UI Assets ===
imgBackground = cv2.imread('Resources/background.png')
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# === Load Precomputed Face Encodings ===
print("Loading Encode File ...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# === YOLO Anti-Spoofing Setup ===
yolo_model = YOLO("../models/n_version_4_75.pt")
confidence_threshold = 0.6
classNames = ["fake", "real"]  # Model predicts whether face is fake or real

# === Camera and App State Variables ===
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

modeType = 0         # Determines which UI mode is shown
counter = 0          # Controls timing for displaying attendance info
id = -1              # ID of the recognized student
studentInfo = {}     # Info fetched from Firebase

# Buffer to avoid sudden fake-real flickers
real_face_buffer = []
buffer_size = 5      # Must see 4+ "real" frames to proceed

# === Main Loop ===
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror camera for natural view

    # Draw webcam into background layout
    frame = imgBackground.copy()
    frame[162:162 + 480, 55:55 + 640] = img
    frame[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    # === Anti-Spoofing Check ===
    real_face_detected = False
    results = yolo_model(img, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = classNames[cls]
                color = (0, 255, 0) if label == "real" else (0, 0, 255)

                # Draw detection box on background layout
                cv2.rectangle(frame, (55 + x1, 162 + y1), (55 + x2, 162 + y2), color, 2)

                # Store prediction in buffer
                real_face_buffer.append(label)
                if len(real_face_buffer) > buffer_size:
                    real_face_buffer.pop(0)

    # Check if most frames in buffer are "real"
    if real_face_buffer.count("real") >= buffer_size - 1:
        real_face_detected = True

    # === If Real Face Detected, Proceed to Face Recognition ===
    if real_face_detected and counter == 0:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    y1, x2, y2, x1 = [val * 4 for val in faceLoc]

                    # Draw green corner box around recognized face
                    corner_color = (0, 255, 0)
                    corner_len = 20
                    thickness = 2
                    x1_disp, y1_disp = 55 + x1, 162 + y1
                    x2_disp, y2_disp = 55 + x2, 162 + y2

                    cv2.line(frame, (x1_disp, y1_disp), (x1_disp + corner_len, y1_disp), corner_color, thickness)
                    cv2.line(frame, (x1_disp, y1_disp), (x1_disp, y1_disp + corner_len), corner_color, thickness)
                    cv2.line(frame, (x2_disp, y1_disp), (x2_disp - corner_len, y1_disp), corner_color, thickness)
                    cv2.line(frame, (x2_disp, y1_disp), (x2_disp, y1_disp + corner_len), corner_color, thickness)
                    cv2.line(frame, (x1_disp, y2_disp), (x1_disp + corner_len, y2_disp), corner_color, thickness)
                    cv2.line(frame, (x1_disp, y2_disp), (x1_disp, y2_disp - corner_len), corner_color, thickness)
                    cv2.line(frame, (x2_disp, y2_disp), (x2_disp - corner_len, y2_disp), corner_color, thickness)
                    cv2.line(frame, (x2_disp, y2_disp), (x2_disp, y2_disp - corner_len), corner_color, thickness)

                    id = studentIds[matchIndex]
                    counter = 1
                    modeType = 1  # Show attendance template

    # === Attendance Logic ===
    if counter != 0:
        if counter == 1:
            # Fetch student info from Firebase
            studentInfo = db.reference(f'Students/{id}').get()
            datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
            secondsElapsed = (datetime.now() - datetimeObject).total_seconds()


            # Prevents marking attendance if it was marked less than a certain time frame (in seconds)
            if secondsElapsed > 30:               # === THIS IS THE 30-SECOND BUFFER LOGIC ==============================
                ref = db.reference(f'Students/{id}')
                studentInfo['total_attendance'] += 1
                ref.child('total_attendance').set(studentInfo['total_attendance'])
                ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            else:
                modeType = 3  # Already marked
                counter = 0  # Skip rest of the attendance flow

        # === Show student info (Mode 1) for 40 frames ===
        if 1 <= counter <= 40:
            modeType = 1
            frame[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            cv2.putText(frame, str(studentInfo["total_attendance"]), (861, 125),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(frame, str(studentInfo["major"]), (1006, 550),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, str(id), (1006, 493),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, str(studentInfo["standing"]), (910, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(frame, str(studentInfo["year"]), (1025, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(frame, str(studentInfo["starting_year"]), (1125, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

            (w, _), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            offset = (414 - w) // 2
            cv2.putText(frame, str(studentInfo['name']), (808 + offset, 445),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

        # === Mode 2: Simple animation, no info ===
        elif 41 <= counter < 70:
            modeType = 2
            frame[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        # After 70 frames, reset everything
        counter += 1
        if counter >= 70:
            counter = 0
            modeType = 0
            studentInfo = {}
            id = -1
            real_face_buffer.clear()
    else:
        # Reset to default state if no attendance being processed
        modeType = 0
        counter = 0

    # === Show Final Frame ===
    cv2.imshow("Face attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
