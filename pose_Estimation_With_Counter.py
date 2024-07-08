import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(start_point, mid_point, end_point):
    # Convert points to NumPy arrays
    start_point = np.array(start_point)
    mid_point = np.array(mid_point)
    end_point = np.array(end_point)
    
    # Calculate the angle using the arctan2 function
    radians = np.arctan2(end_point[1] - mid_point[1], end_point[0] - mid_point[0]) - np.arctan2(start_point[1] - mid_point[1], start_point[0] - mid_point[0])
    # The difference between these two arctan values gives the angle between the 
    # vectors formed by points start_point-mid_point and end_point-mid_point.
    
    # Convert radians to degrees
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Ensure the angle is within the range of 0 to 180 degrees
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize video capture from webcam (index 0)
video_capture = cv2.VideoCapture(0)

# Curl counter variables
rep_counter = 0  # Initialize counter for reps
curl_stage = None  # Variable to track stage (up or down)

# Setup MediaPipe Pose instance
with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while video_capture.isOpened():
        # Read frame from webcam
        ret, frame = video_capture.read()
        # ret is a boolean indicating whether the frame was successfully read.
        # frame contains the captured frame.  
    
        # If frame is read correctly, ret is True
        if not ret:
            print("Failed to grab frame")
            break
        # Convert BGR to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Disable frame writeability to improve performance
      
        # Make pose detection
        results = pose.process(image)
    
        # Recolor frame back to BGR format
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            # Extract landmarks from pose detection results
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates of specific landmarks (right shoulder, right elbow, right wrist)
            right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle between shoulder, elbow, and wrist
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Visualize angle on the frame
            cv2.putText(image, str(angle), 
                        tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Curl counter logic
            if angle > 160:
                curl_stage = "down"
            if angle < 20 and curl_stage == 'down':
                curl_stage = "up"
                rep_counter += 1  # Increment counter on each curl completion
                print(rep_counter)  # Print current count to console
            
        except:
            pass
        
        # Render curl counter information on the frame
        # Setup status box for reps and stage
        cv2.rectangle(image, (0, 0), (250, 100), (245, 117, 16), -1)  # Background rectangle
        
        # Count the reps data
        cv2.putText(image, 'Reps', (15, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(rep_counter), 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Stage data (up or down)
        cv2.putText(image, 'STAGE', (65, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, curl_stage, 
                    (60, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Render pose landmarks and connections on the frame
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                                  landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 117, 66), thickness=2, circle_radius=2),
                                                  connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                                      color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Display the annotated frame with landmarks and counter info
        cv2.imshow('Mediapipe Feed', image)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release webcam and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()
