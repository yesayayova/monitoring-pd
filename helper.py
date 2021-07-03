from darknet import *
import cv2

def darknet_helper(img, width, height):
  network, class_names, class_colors = load_network("cfg/yolov4-tiny.cfg", "cfg/coco.data", "yolov4-tiny.weights")
  width = network_width(network)
  height = network_height(network)

  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  
  return detections, width_ratio, height_ratio