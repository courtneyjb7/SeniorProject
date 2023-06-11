# Help from Mediapipe docs: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
# Head detection code came from tutorial: https://www.youtube.com/watch?v=-toNMaS4SeQ&t=747s
    # with this github: https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py 

import cv2
import mediapipe as mp
import numpy as np
import follow_controller as cnt


max_angle = 135
mid_angle = 90
min_angle = 45
max_arm_slope = 0.5
min_arm_slope = -5
min_head_y = -17
max_head_y = 17
frames_skip = 1
head_frames_skip = 2



def slope(a, b):
    return (a.y - b.y) / (a.x - b.x)


def cap(angle, minV, maxV):
	if angle > maxV:
		return maxV
	elif angle < minV:
		return minV
	return angle

def map_arms(slope):
    s = cap(slope, min_arm_slope, max_arm_slope)
    return round((s - min_arm_slope) * (max_angle - min_angle) / (max_arm_slope - min_arm_slope) + min_angle)
def map_head(y):
    capped_y = cap(y, min_head_y, max_head_y)
    return round((capped_y - min_head_y) * (max_angle - min_angle) / (max_head_y - min_head_y) + min_angle)

if __name__ == "__main__":
	# initialize positions
	cnt.rightArm(min_angle)
	cnt.leftArm(min_angle)
	cnt.head(mid_angle)
	print("Press q to quit.")
	mp_drawing = mp.solutions.drawing_utils
	mp_drawing_styles = mp.solutions.drawing_styles
	mp_pose = mp.solutions.pose
	mp_face_mesh = mp.solutions.face_mesh
	face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

	rightArm_frame = 0
	rightArm_avgAngle = 0
	leftArm_frame = 0
	leftArm_avgAngle = 0
	head_frame = 0
	head_avgAngle = 0

  	# For webcam input:
	capture = cv2.VideoCapture(0)
	with mp_pose.Pose(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as pose:

		while capture.isOpened():
			success, image = capture.read()
			if not success:
				print("Ignoring empty camera frame.")
				break

			# To improve performance, optionally mark the image 
				# as not writeable to pass by reference.
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
		

			### Start Head Code ###
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
					y = angles[1] * 360			

					# Map angle and send to controller (My head code)
					head_frame += 1
					if head_frame == head_frames_skip:
						head_frame = 0
						head_angle = round(head_avgAngle / head_frames_skip)
						print(map_head(head_angle))
						cnt.head(map_head(head_angle))
						head_avgAngle = y
					else:
						head_avgAngle += y
					# print(map_head(y))
					# cnt.head(map_head(y))

			        # See where the user's head tilting
					if y < -10:
						text = "Looking Right"
					elif y > 10:
						text = "Looking Left"
					else:
						text = "Looking Forward"
					
					# Display the nose direction
					nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

					p1 = (int(nose_2d[0]), int(nose_2d[1]))
					p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
					
					cv2.line(image, p1, p2, (255, 0, 0), 3)
					image = cv2.flip(image, 1)
					# Add the text on the image
					# cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
					# cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			### End Head Code ###

			vis_threshold = 0.7
			#### Start Right Arm Code ####	
			try: 
				landmarks = results.pose_landmarks.landmark
				right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
				right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

				#If in frame, find slope
				if (right_elbow.visibility > vis_threshold and 
						right_shoulder.visibility > vis_threshold):
					rightM = slope(right_elbow, right_shoulder)
					angle = map_arms(rightM)
					rightArm_frame += 1

					if rightArm_frame == frames_skip:
						rightArm_frame = 0
						robot_angle = round(rightArm_avgAngle / frames_skip)
						# print(robot_angle)
						cnt.rightArm(robot_angle)
						rightArm_avgAngle = angle
					else:
						rightArm_avgAngle += angle
					cv2.putText(image, "slope: " + str(round(rightM, 3)), 
									tuple(np.multiply([1-right_shoulder.x, right_shoulder.y], [640, 480]).astype(int)), 
									cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
										)   							
			except Exception as e: 
				print(e)
			#### End Right Arm Code ####
				
			#### Start Left Arm Code ####	
			try: 
				landmarks = results.pose_landmarks.landmark
				left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
				left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]	

				if (left_elbow.visibility > vis_threshold and 
						left_shoulder.visibility > vis_threshold):
					leftM = -1 * slope(left_elbow, left_shoulder)
					left_angle = map_arms(leftM)
					leftArm_frame += 1

					if leftArm_frame == frames_skip:
						leftArm_frame = 0
						l_robot_angle = round(leftArm_avgAngle / frames_skip)
						# print(left_angle)
						cnt.leftArm(l_robot_angle)
						leftArm_avgAngle = left_angle
					else:
						leftArm_avgAngle += left_angle
					cv2.putText(image, "slope: " + str(round(leftM, 3)), 
								tuple(np.multiply([1-left_shoulder.x, left_shoulder.y], [640, 480]).astype(int)), 
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
									)  	
										
			except Exception as e: 
				print(e)
			#### End Left Arm Code ####	

			cv2.imshow('MediaPipe Pose', image)

			if cv2.waitKey(5) & 0xFF == ord('q'):
				cnt.rightArm(min_angle)
				cnt.leftArm(min_angle)
				cnt.head(mid_angle)
				break
	capture.release()