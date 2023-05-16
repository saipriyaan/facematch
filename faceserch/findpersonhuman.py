import cv2
import os

# Load pre-trained HOG + SVM detector for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Create output directory if not exists
output_dir = 'allperson'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open input video
cap = cv2.VideoCapture('street.mp4')

# Loop through each frame of the video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect people in the frame
    rects, _ = hog.detectMultiScale(frame)
    
    # Save each person as an individual image file
    for i, (x, y, w, h) in enumerate(rects):
        person_img = frame[y:y+h, x:x+w]
        img_path = os.path.join(output_dir, f'person_{frame_count}_{i}.jpg')
        cv2.imwrite(img_path, person_img)
    
    frame_count += 1

cap.release()
