# Pose Detection and Curl Counter using Mediapipe
This project uses OpenCV and Mediapipe to detect body poses in real-time via a webcam feed and counts the number of curls (bicep curls) performed by a person. The program visually displays the angles of the elbow during the curl and keeps a count of completed curls.

# Features
1.) Real-time Pose Detection: Utilizes Mediapipe's Pose solution to detect body landmarks.
2.) Angle Calculation: Calculates the angle at the joint (in this project calculating the angle at the elbow joint).
3.) Curl Counter: Counts the number of complete curls performed based on the calculated angles.
4.) Visual Feedback: Displays the current angle, curl count, and stage (up/down) on the video feed.
# Prerequisites
Python 3.8,
  OpenCV,
  Mediapipe,
  NumPy,
