import cv2
from realsense_camera import *
from mask_rcnn import *
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
# Load Realsense camera
rs = RealsenseCamera()
mrcnn = MaskRCNN()
while True:
	ret, bgr_frame, depth_frame = rs.get_frame_stream()
	image_height, image_width, _ = bgr_frame.shape
	results = pose.process(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
	get_ratio_x_top = []
	get_ratio_y_top = []
	get_ratio_y_bottom = []
	if results.pose_landmarks:
		for i in range(11, 33):
			if results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility >= 0.9:
				normalizedLandmark = results.pose_landmarks.landmark[i]
				pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, image_width, image_height)
				if pixelCoordinatesLandmark:
					if i == 11:
						pos_x = pixelCoordinatesLandmark[0]
						pos_y = pixelCoordinatesLandmark[1]
						get_ratio_x_top.append(pos_x)
						get_ratio_y_top.append(pos_y)
						print(f"top: {get_ratio_y_top}")
						y_avg_top = sum(get_ratio_y_top, 0.0) / len(get_ratio_y_top)
					if i == 23:
						pos_y2 = pixelCoordinatesLandmark[1]
						get_ratio_y_bottom.append(pos_y2)
						print(f"bottom: {get_ratio_y_bottom}")
						print("######")
						y_avg_bot = sum(get_ratio_y_bottom, 0.0) / len(get_ratio_y_bottom)

	if len(get_ratio_x_top) == 0 or len(get_ratio_y_top) == 0 or len(get_ratio_y_bottom) == 0:
		continue

	print(f"y1 = {y_avg_top}, y2 = {y_avg_bot}")
	x_avg_top = sum(get_ratio_x_top, 0.0) / len(get_ratio_x_top)


	# Get frame in real time from Realsense camera
	# Get object mask
	boxes, classes, contours, centers, id = mrcnn.detect_objects_mask(bgr_frame)
	# Draw object mask
	bgr_frame = mrcnn.draw_object_mask(bgr_frame)

	# Show depth info of the objects
	mrcnn.draw_object_info(bgr_frame, depth_frame, pos_x, pos_y, pos_y2)

	mp_drawing.draw_landmarks(image=bgr_frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
	cv2.imshow("depth frame", depth_frame)
	cv2.imshow("Bgr frame", bgr_frame)

	key = cv2.waitKey(1)
	if key == 27:
		break

rs.release()
cv2.destroyAllWindows()
