import cv2
from deepface import DeepFace

# Load the reference image
reference_image = "reference.png"

# Load the RetinaFace detector
detector = DeepFace.build_model("RetinaFace")

# Load the video file
cap = cv2.VideoCapture("video.mp4")

# Loop through the frames of the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Extract faces from frame
    detected_faces = DeepFace.extract_face(frame, detector_model=detector)

    # Loop through the detected faces
    for i, face in enumerate(detected_faces):
        # Compare the detected face with the reference image
        result = DeepFace.verify(reference_image, face)

        # If face is detected, mark it with a green square and text
        if result['verified']:
            x1, y1, x2, y2 = face['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person Found", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
