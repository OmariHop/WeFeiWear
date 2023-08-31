#https://pysource.com
import cv2
from realsense_camera import *
from mask_rcnn import *

# Load Realsense camera
rs = RealsenseCamera()
mrcnn = MaskRCNN()

while True:
	# Obtaining frames in real time using the real sense camera
	ret, bgr_frame, depth_frame = rs.get_frame_stream()
	cv2.imshow("BGR frame", bgr_frame)
	# Boxes = the box surrounding the identified object 
	# Classes = the type of object being detected
	# Contours = outline of the object 
	# Centers = Point of reference used to get the distance
	boxes, classes, centers = mrcnn.detect_objects_mask(bgr_frame)

	# Printing out information
	
	# Handles getting the distance from the camera to the classified object from the 	
	distances = [depth_frame[cy][cx] for (cx, cy) in centers]
    
    # Print class names alongside their distances
	for class_id, distance in zip(classes, distances):
		class_name = mrcnn.classes[int(class_id)]
		print(f"Detected {class_name} at {distance/10:.2f} cm")


	key = cv2.waitKey(1)
	if key == 27:
		break

rs.release()
cv2.destroyAllWindows()
