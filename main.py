# import darknet functions to perform object detections
from darknet import *
import cv2
import numpy as np
import time
import distance
import helper
import perspective


# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4-tiny.cfg", "cfg/coco.data", "yolov4-tiny.weights")
width = network_width(network)
height = network_height(network)

input_path = "/content/drive/MyDrive/Datasets/scene-5.mp4"
output_path = "/content/drive/MyDrive/Datasets/scene-5-result.mp4"

cap = cv2.VideoCapture(input_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter(output_path ,cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

id_frame = 0

# Read until video is completed
while (cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:
    # Display the resulting frame
    prev_time = time.time()
    detections, width_ratio, height_ratio = helper.darknet_helper(frame, width, height)

    corners_for_polylines = np.array([[420, 325], [940, 325], [1258, 622], [90, 622]])
    corners_x = [420, 940, 90, 1258]
    corners_y = [325, 325, 622, 622]
    corners = perspective.point_corner(corners_x, corners_y)

    n_person = 0
    coordinates = []
    boxes = []

    for label, confidence, bbox in detections:
      if label == 'person':
        left, top, right, bottom = bbox2points(bbox)
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
        box = (left, top, right, bottom)
        #cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        #cv2.circle(frame, (left+(right-left)//2, bottom), 4, (0,255,0), -1)
        x, y = (left+(right-left)//2, bottom)

        if (x > perspective.left_border(y)) and (x < perspective.right_border(y)) and (y > 325) and (y < 622) : 
          coordinates.append((x, y))
          boxes.append(box)
          n_person += 1

    if len(coordinates) > 0 :
      width = np.max(corners_x) - np.min(corners_y)
      height = np.max(corners_x) - np.min(corners_y)
      mat, img = perspective.compute_perspective_transform(corners, width, height, frame)
    
      result = perspective.compute_point_perspective_transformation(mat, coordinates)
      distances = distance.calc_distance(result)

    #print(len(distances))

    object_id = 0
    for box in boxes:
      left, top, right, bottom = box
      if distances[object_id] == 1:
          cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
      else:
          cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
      object_id += 1

    cv2.polylines(frame, [corners_for_polylines], True, (255, 0, 0), 3)
    fps = int(1/(time.time()-prev_time))

    #if id_frame == 1:
     # cv2.imwrite("/content/img_cover.jpg", frame)

    out.write(frame)
    print("writed frame-{} | {} person detected | FPS:{}".format(id_frame, n_person, fps))
    print(distances)
    id_frame += 1

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()