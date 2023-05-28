# Imported from Project Settings> Services Accout> Python
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
# Realtime Database URL is placed in JSON format
firebase_admin.initialize_app(cred, {
    'databaseURL' : "https://faceattendancerealtime-179d4-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# We are creating Employees Database
ref = db.reference('Employees')

data = {
    "076412":
        {
            "name" : "Biplav Poudel",
            "position" : "Secretary",
            "starting_year" : 2080,
            "total_attendance" : 10,
            "year": 1,
            "last_attendance_time" : "2023-05-27 10:54:34"
        },
    "321654":
        {
            "name": "Salina Gurung",
            "position": "Section Officer",
            "starting_year": 2080,
            "total_attendance": 1,
            "year": 1,
            "last_attendance_time": "2023-05-28 12:49:30"
        },
    "852741":
        {
            "name": "Yogesh Bhandari",
            "position": "Office Assistant",
            "starting_year": 2080,
            "total_attendance": 4,
            "year": 1,
            "last_attendance_time": "2023-05-28 13:34:01"
        }
}

# For storing the data
for key, value in data.items():
    ref.child(key).set(value)