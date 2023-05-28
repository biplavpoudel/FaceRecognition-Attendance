import pickle
import numpy as np
import cv2
import os
import cvzone

import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# print(modePathList)
# print(len(imgModeList))

# Now we load the encoding file
print("Loading encoded files...")

file = open("EncodeFile.p", 'rb')
encodeKnownListWithIds = pickle.load(file)
file.close()

encodeKnownList, employeeIds = encodeKnownListWithIds
# print(studentIds)
print("Encoded files loaded successfully.")

while True:
    success, img = cap.read()

    # imgS is just a frame captured by webcam which is refreshed every 1 millisecond??
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # For faces in current frame
    faceCurrentFrame = face_recognition.face_locations(imgS)

    # Now we encode current frame for comparision with stored ones
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)


    imgBackground[162:162+480, 55:55+640] = img    # [height, width] or [y-axes, x-axes]
    imgBackground[44:44+633, 808:808+414] = imgModeList[0]

    # Now we loop through all the encodings and compare with previously stored ones
    # we use zip() to loop two lists together. otherwise we need two loops for each list
    # extracted info from encodeCurrentFrame goes to encodedFace
    # extracted info from faceCurrentFrame goes to faceLocations
    # location info isn't present in encodeCurrentFrame, so we include faceCurrentFrame
    for encodedFace, faceLocations in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeKnownList, encodedFace)
        # we can compare the face distance. Lower the distance, better the match
        faceDist = face_recognition.face_distance(encodeKnownList, encodedFace)

        # print("matches", matches)
        # print("distance", faceDist)
        # Now we get the index of the lower facial_distance values of the four
        matchIndex = np.argmin(faceDist)   # argmin returns the array index with minimum value
        # print("Match Index", matchIndex)

        if matches[matchIndex]:     # if true:
            print("Known Face Detected")
            print("Employees Id: ", employeeIds[matchIndex])
            # we now draw rectangle around the detected face to be fancy 😂😂
            y1, x2, y2, x1 = faceLocations       # this is how face location is mapped. weirdly..
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4      # we reduced the size of image by one-fourth for live-encoding so...
            bbox = 55+x1, 162+y1, x2-x1, y2-y1          # bbox = x1+offset, y1+offset, width, height
            # (55+ x1, 162+y1) = (initial position of actual image in background image in x-axis+ offset position of face in actual image in x-axis, ...)
            # our actual image doesn't start in (0,0) position of imgBackground. Instead, it is (55, 162)
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)  # rt= rectangle thickness, bbox= bounding box


    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
