import cv2
import os
import face_recognition
import pickle

# Importing student images
folderPath = 'Images'
PathList = os.listdir(folderPath)
imgList = []
studentIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    # os.path.join creates the directory for each images as "Images/abc.png"

    # print(path) gives ".png" extension with the imageid i.e."imageid.png"
    # Split the pathname path into a pair(root, ext) such that root + ext == path
    # print(os.path.splitext(path)[0])
    studentIds.append(os.path.splitext(path)[0])

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
encodeKnownListWithIds = [encodeKnownList, studentIds]
print("Encoding Complete")

# pickle file to dump/store the two lists for webcam to compare later on into a file called "file"
file = open("EncodeFile.p", 'wb')
pickle.dump(encodeKnownListWithIds, file)
file.close()
print("File Saved")