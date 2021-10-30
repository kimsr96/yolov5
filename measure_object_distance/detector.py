import cv2
import mediapipe as mp
import re
from realsense_depth import *
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

PART = {
    0 : "nose",
    1 : "left_eye_inner",
    2 : "left_eye",
    3 : "left_eye_outer",
    4 : "right_eye_inner",
    5 : "right_eye",
    6 : "right_eye_outer",
    7 : "left_ear",
    8 : "right_ear",
    9 : "mouth_left",
    10 : "mouth_right",
    11 : "left_shoulder",
    12 : "right_shoulder",
    13 : "left_elbow",
    14 : "right_elbow",
    15 : "left_wrist",
    16 : "right_wrist",
    17 : "left_pinky",
    18 : "right_pinky",
    19 : "left_index",
    20 : "right_index",
    21 : "left_thumb",
    22 : "right_thumb",
    23 : "left_hip",
    24 : "right_hip",
    25 : "left_knee",
    26 : "right_knee",
    27 : "left_ankle",
    28 : "right_ankle",
    29 : "left_heel",
    30 : "right_heel",
    31 : "left_foot_index",
    32 : "right_foot_index"
}  

def height_(chunk, distance_chunk):
    vec1 = np.array([chunk[1][0], chunk[1][1], distance_chunk[1]])
    vec2 = np.array([chunk[29][0], chunk[29][1], distance_chunk[29]])
    innerAB = np.dot(vec1, vec2)
    AB = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    ang = np.arccos(innerAB/AB)
    sq = ((np.linalg.norm(vec1) ** 2) + (np.linalg.norm(vec2) ** 2) - 2 * np.linalg.norm(vec1) *np.linalg.norm(vec2) * math.cos(ang)) ** 0.5
    return sq

dc = DepthCamera()
distance_chunk = [None for i in range(33)]
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while True:
    ret, depth_frame, color_frame = dc.get_frame()

    image = cv2.cvtColor(cv2.flip(color_frame, 1), cv2.COLOR_BGR2RGB)
    depth_frame = cv2.flip(depth_frame, 1)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # global variables
    
    if(results.pose_landmarks != None):
        lm = results.pose_landmarks.landmark
        lm_str = str(lm)
        lm_num = re.findall(r'-?\d+\.\d+', lm_str)
        lm_num = [float(s) for s in lm_num]
        lm_num_chunk = [lm_num[i : i+4] for i in range(0, len(lm_num), 4)]
        
        height, width, channel = image.shape
        x_max, x_min, y_max, y_min = (0, 1000, 0, 1000)
        #boundary  = [ max(lm_num_chunk[:][0]), max(lm_num_chunk[:][1]), max(lm_num_chunk[:][2])]
        for i in range(0, len(lm_num_chunk)):
            if(lm_num_chunk[i][3] < 0.8):
                lm_num_chunk[i][0] = None
                lm_num_chunk[i][1] = None
                lm_num_chunk[i][2] = None
            else:
                x, y = (lm_num_chunk[i][0]*width, lm_num_chunk[i][1]*height)
                if(x == None or y == None):
                    continue
                if(x_max < x):
                    x_max = int(x)
                if (y_max < y):
                    y_max = int(y)
                if (x_min > x):
                    x_min = int(x)
                if (y_min > y):
                    y_min = int(y)
                #print(results.pose_landmarks)
                #print(lm_num_chunk)
                if (int(x) >= 1280 or int(y) >= 720 or int(x) < 0 or int(y) < 0):
                    distance = None
                    distance_chunk[i] = None
                    continue
                else: 
                    if (depth_frame[int(y), int(x)] < 0):
                        continue
                    distance = depth_frame[int(y), int(x)]
                    distance_chunk[i] = distance
                    #print(distance)
                #cv2.putText(image, distance, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 2)
                #print(x_max, x_min)
            
        if (lm_num_chunk[1][0] != None and lm_num_chunk[29][0] != None and
                distance_chunk[1] != 0 and distance_chunk[29] != 0):
            print(height_(lm_num_chunk, distance_chunk))
            
        #print(distance_chunk)
        #distance = depth_frame[lm_num_chunk[23][0]*width, lm_num_chunk[23][1]*height]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    
    
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)

    #cv2.imshow("depth frame", depth_frame)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break






#def ratio(chunk):
    