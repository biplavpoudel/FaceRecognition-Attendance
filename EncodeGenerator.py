import cv2
import os
import face_recognition
import pickle


# Imported from Project Settings> Services Accout> Python
# for uploading image to the database while encoding
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

# Importing student images
folderPath = 'Images'
PathList = os.listdir(folderPath)
imgList = []
employeeIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    # os.path.join creates the directory for each images as "Images/abc.png"

    # print(path) gives ".png" extension with the imageid i.e."imageid.png"
    # Split the pathname path into a pair(root, ext) such that root + ext == path
    # print(os.path.splitext(path)[0])
    employeeIds.append(os.path.splitext(path)[0])

# print(len(imgList))
# print(imgList)
# print(studentIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # OpenCV uses BGR colorspace but face_recognition uses RGB, so we need conversion
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

# If lots of images, it may take a while, so we print Encoding started/completed messages
print("Encoding has been initiated...")
encodeKnownList = findEncodings(imgList)
encodeKnownListWithIds = [encodeKnownList, employeeIds]
print("Encoding Complete")

# pickle file to dump/store the two lists for webcam to compare later on into a file called "file"
file = open("EncodeFile.p", 'wb')
pickle.dump(encodeKnownListWithIds, file)
file.close()
print("File Saved")