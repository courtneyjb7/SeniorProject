#Base code from Mediapipe docs: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
import cv2
import mediapipe as mp
import numpy as np
import controller as cnt

# could check if arm is straight -> angle at elbow (from shoulder to elbow to wrist) is close to 180
print("Press q to quit.")
cnt.rightArm("down")
cnt.leftArm("down")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def slope(a, b):
    return (a.y - b.y) / (a.x - b.x)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
  

    image = cv2.flip(image, 1)

    vis_threshold = 0.7
    try:
      landmarks = results.pose_landmarks.landmark
      left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
      left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

      #If in frame, find slope
      if (left_elbow.visibility > vis_threshold and left_shoulder.visibility > vis_threshold):
        leftM = slope(left_elbow, left_shoulder)
        if leftM > 0:
          messageL = "left arm DOWN"
          cnt.leftArm("down")
        else:
          messageL = "left arm UP"
          cnt.leftArm("up")
          
        cv2.putText(image, messageL, 
                        tuple(np.multiply([0, 0.1], [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                            )
        cv2.putText(image, "slope: " + str(round(leftM, 3)), 
                        tuple(np.multiply([1-left_shoulder.x, left_shoulder.y], [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
    except:
      pass

    try:
      landmarks = results.pose_landmarks.landmark
      right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
      right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

      #If in frame, find slope
      if (right_elbow.visibility > vis_threshold and right_shoulder.visibility > vis_threshold):
        rightM = slope(right_elbow, right_shoulder)
        if rightM > 0:
          messageR = "right arm UP"
          cnt.rightArm("up")
        else:
          messageR = "right arm DOWN"
          cnt.rightArm("down")
        cv2.putText(image, messageR, 
                        tuple(np.multiply([0.5, 0.1], [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                            )
        cv2.putText(image, "slope: " + str(round(rightM, 3)), 
                        tuple(np.multiply([1-right_shoulder.x, right_shoulder.y], [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
    except:
      pass
    

    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()