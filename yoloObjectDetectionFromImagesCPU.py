import cv2
import numpy as np
import os  # Ensure the os module is imported

# Load YOLOv3 model
yolo = cv2.dnn.readNet("D:/IIT-Goa/Sem-7/BTP/YOLO_Defect-Detection/yolov3.weights", 
                        "D:/IIT-Goa/Sem-7/BTP/YOLO_Defect-Detection/yolov3.cfg")

# Load class names
classes = []
with open("obj.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Get output layer names
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers().flatten()]

colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

# Folder containing the images
image_folder = "Dataset/square_images"
output_dir = "output_images"

# Loop over each image in the folder
for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg"):
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        print(f"Processing image: {img_path}")

        # Resize image to 416x416 for YOLO input
        img = cv2.resize(img, (416, 416))
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo.setInput(blob)
        outputs = yolo.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:  # Lowered confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Perform non-maxima suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels
        if len(indexes) > 0:
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
                    cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 8, colorRed, 8)

            # Save the output image
            output_path = os.path.join(output_dir, image_file)
            try:
                cv2.imwrite(output_path, img)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed to save {output_path}: {e}")
        else:
            print(f"No detections in {image_file}")

print("Processing complete. Check the output_images folder for results.")


# import cv2
# import numpy as np
# import os

# # Load YOLOv3 model
# # yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# yolo = cv2.dnn.readNet("D:/IIT-Goa/Sem-7/BTP/YOLO_Defect-Detection/yolov3.weights", 
#                         "D:/IIT-Goa/Sem-7/BTP/YOLO_Defect-Detection/yolov3.cfg")

# # Load class names
# classes = []
# with open("obj.names", "r") as file:
#     classes = [line.strip() for line in file.readlines()]

# # Get output layer names
# layer_names = yolo.getLayerNames()
# output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers().flatten()]

# colorRed = (0, 0, 255)
# colorGreen = (0, 255, 0)

# # Folder containing the images
# image_folder = "Dataset/square_images"

# # Loop over each image in the folder
# for image_file in os.listdir(image_folder):
#     if image_file.endswith(".jpg"):
#         # Read the image
#         img_path = os.path.join(image_folder, image_file)
#         img = cv2.imread(img_path)
#         height, width, channels = img.shape

#         # Detecting objects
#         blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#         yolo.setInput(blob)
#         outputs = yolo.forward(output_layers)

#         class_ids = []
#         confidences = []
#         boxes = []

#         for output in outputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#                 if confidence > 0.5:
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)

#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#         # Perform non-maxima suppression
#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#         # Draw bounding boxes and labels
#         for i in range(len(boxes)):
#             if i in indexes:
#                 x, y, w, h = boxes[i]
#                 label = str(classes[class_ids[i]])
#                 cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
#                 cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 8, colorRed, 8)

#         # # Save the output image
#         # output_path = os.path.join("output_images", image_file)
#         # cv2.imwrite(output_images, img)
#         output_dir = "output_images"

#         # Ensure the output_images directory exists
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         # Save the output image
#         output_path = os.path.join(output_dir, image_file)
#         cv2.imwrite(output_path, img)
# print("Processing complete. Check the output_images folder for results.")



#------------------------------------------------------------------------------------------------------------
# import cv2
# import numpy as np

# yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# classes = []

# with open("obj.names", "r") as file:
#     classes = [line.strip() for line in file.readlines()]
# # layer_names = yolo.getLayerNames()
# # output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]
# layer_names = yolo.getLayerNames()
# output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers().flatten()]

# colorRed = (0,0,255)
# colorGreen = (0,255,0)

# # #Loading Images
# name = "image.jpg"
# img = cv2.imread(name)
# height, width, channels = img.shape

# # # Detecting objects
# blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# yolo.setInput(blob)
# outputs = yolo.forward(output_layers)

# class_ids = []
# confidences = []
# boxes = []
# for output in outputs:
#     for detection in output:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#         if confidence > 0.5:
#             center_x = int(detection[0] * width)
#             center_y = int(detection[1] * height)
#             w = int(detection[2] * width)
#             h = int(detection[3] * height)

#             x = int(center_x - w / 2)
#             y = int(center_y - h / 2)

#             boxes.append([x, y, w, h])
#             confidences.append(float(confidence))
#             class_ids.append(class_id)

# indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# for i in range(len(boxes)):
#     if i in indexes:
#         x, y, w, h = boxes[i]
#         label = str(classes[class_ids[i]])
#         cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
#         cv2.putText(img, label, (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 8, colorRed, 8)


# #cv2.imshow("Image", img)
# cv2.imwrite("output.jpg",img)
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()