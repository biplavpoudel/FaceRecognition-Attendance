import pickle
import numpy as np
import cv2
import os
import cvzone
import face_recognition

# for real-time database update
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
# Realtime Database URL is placed in JSON format
firebase_admin.initialize_app(cred, {
    'databaseURL' : "https://faceattendancerealtime-179d4-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket' : "faceattendancerealtime-179d4.appspot.com"
})

bucket = storage.bucket()

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

modeType = 0  # mode: active, info, marked, already marked
counter = 0  # to count no of matched image frames
id = -1  # for storing matched employee id
imgEmployee = []


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
    imgBackground[44:44+633, 808:808+414] = imgModeList[modeType]

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
            # print("Known Face Detected")
            # print("Employees Id: ", employeeIds[matchIndex])
            # we now draw rectangle around the detected face to be fancy 😂😂
            y1, x2, y2, x1 = faceLocations       # this is how face location is mapped. weirdly..
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4      # we reduced the size of image by one-fourth for live-encoding so...
            bbox = 55+x1, 162+y1, x2-x1, y2-y1          # bbox = x1+offset, y1+offset, width, height
            # (55+ x1, 162+y1) = (initial position of actual image in background image in x-axis+ offset position of face in actual image in x-axis, ...)
            # our actual image doesn't start in (0,0) position of imgBackground. Instead, it is (55, 162)
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)  # rt= rectangle thickness, bbox= bounding box

            id = employeeIds[matchIndex]     # id of face-matched employee

            # if known face detected and counter is not set to 1, we set it to 1
            if counter == 0:
                counter = 1
                modeType = 1

    if counter != 0:
        if counter == 1:
            # this is the part where we download the matched employee id
            EmployeeInfo = db.reference(f'Employees/{id}').get()
            print(EmployeeInfo)

            # get image from storage
            blob = bucket.get_blob(f'Images/{id}'.png)
            array = np.frombuffer(blob.download_as_string(), np.unint8)
            imgEmployee = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

        cv2.putText(imgBackground,str(EmployeeInfo['total_attendance']), (861,125), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.putText(imgBackground, str(EmployeeInfo['position']), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5,(255, 255, 255), 1)
        cv2.putText(imgBackground, str(id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5,(100, 100, 100), 1)
        cv2.putText(imgBackground, str(EmployeeInfo['rank']), (910, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(imgBackground, str(EmployeeInfo['year']), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,(100, 100, 100), 1)
        cv2.putText(imgBackground, str(EmployeeInfo['starting_year']), (1125, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,(100, 100, 100), 1)

        # to center position the data "name"
        # if total width for name placeholder is "TOTAL" and width of my name is "width",
        # then to center my "name", the formula for starting position is: (TOTAL - width)/2
        (w, h), _ = cv2.getTextSize(EmployeeInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        offset = (414 - w)//2             # total width of Mode 2 image is 414 and // is floor division
        # using normal division gives typecasting errors
        cv2.putText(imgBackground, str(EmployeeInfo['name']), (808+offset, 445), cv2.FONT_HERSHEY_COMPLEX, 1,(50, 50, 50), 1)

        counter += 1


    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
