# Importing necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the Euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute and return the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate the Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mou):
    # Compute the Euclidean distances between the horizontal mouth landmarks
    X = dist.euclidean(mou[0], mou[6])
    # Compute the Euclidean distances between the vertical mouth landmarks
    Y1 = dist.euclidean(mou[2], mou[10])
    Y2 = dist.euclidean(mou[4], mou[8])
    # Take the average of the vertical distances
    Y = (Y1 + Y2) / 2.0
    # Compute and return the MAR
    mar = Y / X
    return mar

# Initialize video capture
camera = cv2.VideoCapture(0)
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Define constants for EAR and MAR thresholds
EYE_AR_THRESH = 0.25  # Threshold for eye aspect ratio
EYE_AR_CONSEC_FRAMES = 48  # Number of consecutive frames for drowsiness detection
MOU_AR_THRESH = 0.75  # Threshold for mouth aspect ratio

# Initialize counters and statuses
COUNTER = 0
yawnStatus = False
yawns = 0

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Get indexes for eyes and mouth from facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Start the video capture loop
while True:
    # Read a frame from the camera
    ret, frame = camera.read()
    frame = imutils.resize(frame, width=640)  # Resize frame for consistency
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    prev_yawn_status = yawnStatus  # Store previous yawn status

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Process each detected face
    for rect in rects:
        # Get facial landmarks and convert them to NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract coordinates for eyes and mouth
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        # Calculate EAR for both eyes and MAR for the mouth
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouEAR = mouth_aspect_ratio(mouth)

        # Average the EAR for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Compute convex hulls for visualization
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        # Draw contours for eyes and mouth on the frame
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Check if EAR is below the threshold (indicating closed eyes)
        if ear < EYE_AR_THRESH:
            COUNTER += 1  # Increment the counter
            cv2.putText(frame, "Eyes Closed ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Trigger drowsiness alert if eyes are closed for sufficient frames
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0  # Reset the counter
            cv2.putText(frame, "Eyes Open ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display EAR on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check if MAR exceeds the threshold (indicating yawning)
        if mouEAR > MOU_AR_THRESH:
            cv2.putText(frame, "Yawning ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawnStatus = True
            output_text = "Yawn Count: " + str(yawns + 1)
            cv2.putText(frame, output_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            yawnStatus = False

        # Update yawn count if a yawn is completed
        if prev_yawn_status == True and yawnStatus == False:
            yawns += 1

        # Display MAR on the frame
        cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Lusip Project @ Swarnim", (370, 470), cv2.FONT_HERSHEY_COMPLEX, 0.6, (153, 51, 102), 1)

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

# Clean up and release resources
cv2.destroyAllWindows()
camera.release()
