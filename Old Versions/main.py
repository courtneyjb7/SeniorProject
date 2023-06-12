# Help from Mediapipe docs: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
# Head detection code came from tutorial: https://www.youtube.com/watch?v=-toNMaS4SeQ&t=747s
    # with this github: https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py 

import cv2
import mediapipe as mp
import numpy as np
import controller as cnt

# values to use as input to neural network: leftM, rightM, y

# could check if arm is straight -> angle at elbow (from shoulder to elbow to wrist) is close to 180
print("Press q to quit.")
cnt.rightArm("down")
cnt.leftArm("down")
cnt.head("forward")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    face_results = face_mesh.process(image)


    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
  

    

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          

            # See where the user's head tilting
            if y < -10:
                text = "Looking Right"
                cnt.head("right")
            elif y > 10:
                text = "Looking Left"
                cnt.head("left")
            # elif x < -10:
            #     text = "Looking Down"
            # elif x > 10:
            #     text = "Looking Up"
            else:
                text = "Forward"
                cnt.head("forward")

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            image = cv2.flip(image, 1)
            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        

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
                        tuple(np.multiply([0, 0.2], [640, 480]).astype(int)), 
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
                        tuple(np.multiply([0, 0.3], [640, 480]).astype(int)), 
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