import threading
import cv2
from deepface import DeepFace


cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

counter=0

facematch=False
referenceimg=cv2.imread("reference.png")

def check_face():
    global facematch
    try:
        if DeepFace.verify(frame,referenceimg.copy())["verified"]:
            facematch=True
        else:
            facematch=False

    except ValueError:
        facematch=False
while True:
    ret,frame=cap.read()
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face(),args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter+=1
        if facematch:
            cv2.putText(frame,"Match found.",(20,450),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,"Searching>>",(20,450),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,0,255),3)
        
        cv2.imshow("video",frame)




        
    key=cv2.waitKey(1)
    if key==ord("q"):
        break

cv2.destroyAllWindows()



