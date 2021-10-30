import cv2
import numpy as np
import math



class MaskRCNN:
    def __init__(self):
        # Loading Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                            "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Generate random colors
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        # Conf threshold
        self.detection_threshold = 0.7
        self.mask_threshold = 0.3

        self.classes = []
        with open("dnn/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []
        self.obj_id = []

        # Distances
        self.distances = []

    def detect_objects_mask(self, bgr_frame):
        blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
        self.net.setInput(blob)

        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        #print(boxes)
        # Detect objects
        frame_height, frame_width, _ = bgr_frame.shape
        detection_count = boxes.shape[2]

        # Object Boxes
        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []
        self.obj_id = []

        for i in range(3):
            print(i)
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            color = self.colors[int(class_id)]
            if score < self.detection_threshold:
                continue

            # Get box Coordinates
            x = int(box[3] * frame_width)
            y = int(box[4] * frame_height)
            x2 = int(box[5] * frame_width)
            y2 = int(box[6] * frame_height)
            self.obj_boxes.append([x, y, x2, y2])

            cx = (x + x2) // 2
            cy = (y + y2) // 2
            self.obj_centers.append((cx, cy))

            # append class
            self.obj_classes.append(class_id)
            self.obj_id.append(i)
            # Contours
            # Get mask coordinates
            # Get the mask
            mask = masks[i, int(class_id)]
            roi_height, roi_width = y2 - y, x2 - x
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.obj_contours.append(contours)

        return self.obj_boxes, self.obj_classes, self.obj_contours, self.obj_centers, self.obj_id

    def draw_object_mask(self, bgr_frame):
        # loop through the detection
        for box, class_id, contours in zip(self.obj_boxes, self.obj_classes, self.obj_contours):
            x, y, x2, y2 = box
            roi = bgr_frame[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape
            color = self.colors[int(class_id)]

            roi_copy = np.zeros_like(roi)

            for cnt in contours:
                # cv2.f(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                cv2.drawContours(roi, [cnt], - 1, (int(color[0]), int(color[1]), int(color[2])), 3)
                cv2.fillPoly(roi_copy, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                roi = cv2.addWeighted(roi, 1, roi_copy, 0.5, 0.0)
                bgr_frame[y: y2, x: x2] = roi
        return bgr_frame

    def draw_object_info(self, bgr_frame, depth_frame, x_avg_top, y_avg_top, y_avg_bot):
        # loop through the detection
        if x_avg_top != None and y_avg_top != None and y_avg_bot != None:
            for box, class_id, obj_center,idx in zip(self.obj_boxes, self.obj_classes, self.obj_centers, self.obj_id):
                x, y, x2, y2 = box
                color = self.colors[int(class_id)]
                color = (int(color[0]), int(color[1]), int(color[2]))
                #cx, cy = obj_center
                cx = int(x_avg_top)
                cy = int((y_avg_top + y_avg_bot) / 2)
                depth_mm = depth_frame[cy, cx]

                cx = int(x_avg_top)
                y  = int(y_avg_top)
                y2 = int(y_avg_bot)
                print(cx, y, y2)

                ang_h1 = (cx - 320) / 320 * (86 / 2)
                ang_v1 = (y - 240) / 240 * (57 / 2)
                ang1 =  round(math.sqrt(ang_h1 ** 2 + ang_v1 **2), 1)

                ang_v2 = (y2 - 240) / 240 * (57 / 2)
                ang2 =  round(math.sqrt(ang_h1 ** 2 + ang_v2 **2), 1)

                ang = abs(ang1) + abs(ang2)
                print(f"ang1, ang2 = {ang1}, {ang2}")
                
                d_top = depth_frame[y, cx] / 10
                d_bottom = depth_frame[y2, cx] / 10
                print(f"Distance top= {d_top} bottom = {d_bottom}")

                #height = (d_top ** 2) + (d_bottom ** 2) - (2 * d_top * d_bottom * math.cos(ang))
                Height = (d_top + d_bottom) / depth_mm
                print(f"Height = {(d_top + d_bottom) / depth_mm}")

                cv2.line(bgr_frame, (cx, y), (cx, y2), color, 1)
                cv2.line(bgr_frame, (x, cy), (x2, cy), color, 1)

                class_name = "person"                
                cv2.rectangle(bgr_frame, (x, y), (x + 250, y + 70), color, -1)
                #cv2.putText(bgr_frame, "{},{},{},{}".format(x,y,x2,y2), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
                #cv2.putText(bgr_frame, "{},{}".format(idx, class_name), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
                #cv2.putText(bgr_frame, "{} cm".format(depth_mm  / 10), (x + 5, y + 60), 0, 1.0, (255, 255, 255), 2)
                cv2.putText(bgr_frame, "{}".format(Height), (x + 5, y + 60), 0, 1.0, (255, 255, 255), 2)
                cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)


        return bgr_frame





