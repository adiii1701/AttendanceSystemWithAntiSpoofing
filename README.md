Real-Time Face Recognition Attendance System
This project is a real-time face recognition-based attendance system built using Python, OpenCV, and Firebase. It automates the attendance process by detecting and recognizing faces from a webcam feed, then marking attendance in a Firebase Realtime Database. This system is designed for educational institutions looking to modernize their attendance tracking methods.

The system uses the face_recognition library to identify faces with high accuracy. It captures frames from the webcam, detects faces, and compares them to known encodings stored in a serialized file using Python’s pickle module. When a match is found, the student’s information—such as name, ID, major, and attendance count—is retrieved from Firebase.

To prevent duplicate attendance within a short time, the system checks if at least 30 seconds have passed since the student's last recorded attendance. If so, it updates the total attendance and records the new timestamp.

A visually engaging interface is created using OpenCV and cvzone, featuring a custom background and multiple visual modes. These modes indicate different system states such as loading, attendance marked, already marked, or attendance denied (if anti-spoofing is added). The interface displays the student's data and provides real-time visual feedback to the user.

The system ensures quick and contactless attendance marking, improving both accuracy and efficiency. It eliminates the need for manual attendance or ID cards, reducing the chances of proxy attendance.

This project can be expanded further by integrating anti-spoofing techniques to detect fake or printed faces and by leveraging GPU acceleration to improve real-time performance. It can also be adapted for use in workplaces, events, or secure entry systems.

In summary, this face recognition attendance system offers a reliable, scalable, and user-friendly solution for automating attendance using computer vision and cloud integration.

REQUIRED INSTALLATION:
pip install opencv-python face_recognition firebase-admin ultralytics numpy cmake


make sure you also install Visual C++ Build Tools from:
👉 https://visualstudio.microsoft.com/visual-cpp-build-tools/



File strusture for reference:



![image](https://github.com/user-attachments/assets/4e589619-40ae-412c-a141-be843f5c93ea)
