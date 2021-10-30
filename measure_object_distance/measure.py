import cv2
from realsense_camera import *
from mask_rcnn import *

rs = RealsenseCamera()
mrcnn = MaskRCNN()

while True:
    #Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()
    
    #Get object mask
    boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

    #Draw object mask
    bgr_frame = mrcnn.draw_object_mask(bgr_frame)

    print("boxes", boxes)
    print("classes", classes)
    #print("contours", contours)
    print("centers", centers)

    mrcnn.detect_objects_mask(bgr_frame)
    distance_mm = depth_frame[100,250]
    cv2.imshow("Bgr frame", bgr_frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
