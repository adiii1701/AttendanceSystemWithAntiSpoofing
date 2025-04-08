import os
import pickle
import cv2
import face_recognition
import cvzone
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase app using the provided service account key and set the database URL.
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-2e6b8-default-rtdb.firebaseio.com/"
})

# Initialize camera capture (webcam) with specific width and height settings.
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Read the main background image that will be used as the canvas for UI.
imgBackground = cv2.imread('Resources/background.png')

# Load the different mode images from the "Resources/Modes" folder.
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# Now imgModeList holds all mode images (like loading screen, attendance info, etc.)

# Load the pre-computed face encodings and corresponding student IDs from a pickle file.
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
# Unpack the list to get known encodings and student IDs.
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encode File Loaded")

# Initialize control variables: modeType for UI mode, counter for timing events, and an id placeholder.
modeType = 0         # UI display mode; 0 might be default, 1 - loading, 2 - info, etc.
counter = 0          # Counter to control the duration of UI state transitions.
id = -1              # Initial student ID (will update when a known face is detected).

# Main loop to continuously read frames from the camera.
while True:
    success, img = cap.read()  # Capture a frame from the camera.
    img = cv2.flip(img, 1)     # Flip the frame horizontally for a mirror-like effect.

    # Create a smaller version of the image for faster face recognition processing.
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV default) to RGB.

    # Detect face locations and compute face encodings in the smaller image.
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Overlay the captured frame onto the main background image.
    # Here the live feed is placed at a specific location on the background.
    imgBackground[162:162 + 480, 55:55 + 640] = img
    # Overlay the UI mode image on the background.
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    # If faces are detected in the frame.
    if faceCurFrame:
        # Loop through each detected face and its encoding.
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            # Compare the detected face encoding to known encodings and calculate distances.
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # Get the index of the best match with the smallest distance.
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                # If a known face is detected, draw a rectangle around the face on the background.
                y1, x2, y2, x1 = faceLoc
                # Since face detection was done on the resized image, rescale the coordinates.
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                # Calculate bounding box coordinates relative to the background's placement.
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                # Retrieve the student ID corresponding to the matched face.
                id = studentIds[matchIndex]

                # If this is the first frame when the face has been detected, trigger a UI update.
                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1  # Start the counter for this detection event.
                    modeType = 1  # Change mode to indicate loading (or similar state).

        # If the counter is non-zero, process the detected face information.
        if counter != 0:
            if counter == 1:
                # Get student information from Firebase database using the detected student id.
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)

                # Retrieve the last attendance time from database and convert to a datetime object.
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                    "%Y-%m-%d %H:%M:%S")

                # Calculate the elapsed seconds since the last attendance record.
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)
                # Update attendance if more than 30 seconds have passed since the last record.
                if secondsElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1  # Increment total attendance count.
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    # If less than 30 seconds since last attendance, set a different mode to indicate quick re-scan.
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            # If the mode is not the "already processed" mode.
            if modeType != 3:
                # Change the mode based on the counter value to create transitions in the user interface.
                if 10 < counter < 20:
                    modeType = 2

                # Refresh the UI mode section of the background.
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                # While the counter is low, overlay the student's details on the background.
                if counter <= 10:
                    # Display total attendance count.
                    cv2.putText(imgBackground, str(studentInfo["total_attendance"]), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    # Display student's major.
                    cv2.putText(imgBackground, str(studentInfo["major"]), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    # Display student ID.
                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    # Display student's standing.
                    cv2.putText(imgBackground, str(studentInfo["standing"]), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    # Display student's year.
                    cv2.putText(imgBackground, str(studentInfo["year"]), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    # Display student's starting year.
                    cv2.putText(imgBackground, str(studentInfo["starting_year"]), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    # Center the student's name in the given UI box.
                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                # Increment the counter to move through the UI states.
                counter += 1

                # Reset the UI and counter after a set number of iterations.
                if counter >= 20:
                    counter = 0
                    modeType = 0  # Return to default UI mode.
                    studentInfo = []  # Clear current student's info.
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    else:
        # If no face is detected in the current frame, reset UI state variables.
        modeType = 0
        counter = 0

    # Display the final composite image on the screen.
    cv2.imshow("Face attendance", imgBackground)
    cv2.waitKey(1)
