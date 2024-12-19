Drowsiness and Yawn Detection System
This project implements a real-time system for detecting drowsiness and yawning using computer vision and facial landmark detection. It uses the dlib library to identify facial landmarks and computes the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to monitor a person's state.

Features
Drowsiness Detection:

Detects when the eyes are closed for a prolonged period (based on EAR).
Triggers a drowsiness alert if the eyes remain closed for a specified number of consecutive frames.
Yawning Detection:

Identifies yawning by calculating the MAR.
Counts and displays the number of yawns detected.
Real-Time Feedback:

Provides visual cues on the screen, such as contours around the eyes and mouth.
Displays EAR and MAR values dynamically on the video feed.
How It Works
Eye Aspect Ratio (EAR)
The EAR is calculated using the Euclidean distance between vertical and horizontal eye landmarks. If the EAR drops below a threshold for a certain number of consecutive frames, it indicates that the eyes are closed, and the system triggers a drowsiness alert.

Mouth Aspect Ratio (MAR)
The MAR is computed as the ratio of the average vertical distance to the horizontal distance of the mouth landmarks. If the MAR exceeds a defined threshold, it indicates yawning, and the yawn count is incremented.
