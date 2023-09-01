import cv2
import numpy as np

class MaskRCNN:
    def __init__(self):
        # Loading Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                                 "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        out_names = self.net.getUnconnectedOutLayersNames()
        print(out_names)

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Conf threshold
        self.detection_threshold = 0.7

        self.classes = []
        with open("dnn/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []

    def detect_objects_mask(self, bgr_frame):
        blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
        self.net.setInput(blob)

        boxes, masks = self.net.forward(["detection_out_final",  "detection_masks"])

        frame_height, frame_width, _ = bgr_frame.shape
        detection_count = boxes.shape[2]

        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []

        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]

            if score < self.detection_threshold:
                continue

            x = int(box[3] * frame_width)
            y = int(box[4] * frame_height)
            x2 = int(box[5] * frame_width)
            y2 = int(box[6] * frame_height)
            self.obj_boxes.append([x, y, x2, y2])

            cx = (x + x2) // 2
            cy = (y + y2) // 2
            self.obj_centers.append((cx, cy))
            self.obj_classes.append(class_id)

        return self.obj_boxes, self.obj_classes, self.obj_centers

    def get_object_info(self, depth_frame):
        obj_info = []

        for box, class_id, obj_center in zip(self.obj_boxes, self.obj_classes, self.obj_centers):
            cx, cy = obj_center
            depth_mm = depth_frame[cy, cx]
            class_name = self.classes[int(class_id)]
            obj_info.append({
                'class_name': class_name,
                'distance_cm': depth_mm / 10
            })

        return obj_info
