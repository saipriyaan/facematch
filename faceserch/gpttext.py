import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

counter = 0
facematch = False
referenceimg = cv2.imread("reference.png")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def check_face():
    global facematch
    try:
        if DeepFace.verify(frame, referenceimg.copy())["verified"]:
            facematch = True
        else:
            facematch = False
    except ValueError:
        facematch = False
        
while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face(), args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1
        if facematch:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Searching>>", (20, 450), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.imshow("video", frame)
        
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
