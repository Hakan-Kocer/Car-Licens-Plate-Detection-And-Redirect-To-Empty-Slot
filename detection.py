import cv2
import numpy as np

def detection(img):
    img_copy = img.copy()
    width = img.shape[1]
    height = img.shape[0]
    
    # Lists to store the best detections
    best_confidence = []
    best_start_x = []
    best_start_y = []
    best_end_x = []
    best_end_y = []
    
    # Prepare image blob
    img_blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    labels = ["plate"]
    colors = [0, 255, 255]
    
    # Load model
    model = cv2.dnn.readNetFromDarknet("plate_yolov4.cfg", darknetModel="plate_yolov4_best.weights")
    layers = model.getLayerNames()
    output_layer = [layers[layer[0] - 1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(img_blob)
    detection_layers = model.forward(output_layer)
    
    # Process detection results
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            # Check if the detection is confident enough
            if confidence > 0.30:
                label = layers[predicted_id]
                bounding_box = object_detection[0:4] * np.array([width, height, width, height])
                (box_center_x, box_center_y, box_w, box_h) = bounding_box.astype("int")
                start_x = int(box_center_x - (box_w / 2))
                start_y = int(box_center_y - (box_h / 2))
                
                # Append best detection data
                best_start_x.append(start_x)
                best_start_y.append(start_y)
                end_x = start_x + box_w
                end_y = start_y + box_h
                best_end_x.append(end_x)
                best_end_y.append(end_y)
                best_confidence.append(float(confidence))

    # Find the detection with the highest confidence
    max_index = np.argmax(best_confidence)
    
    # Draw rectangle around the detected object
    cv2.rectangle(
        img_copy,
        (best_start_x[max_index], best_start_y[max_index]),
        (best_end_x[max_index], best_end_y[max_index]),
        [255, 0, 0],
        1
    )
    
    # Crop the detected area
    cropped = img_copy[
        best_start_y[max_index]:best_end_y[max_index] + 1,
        best_start_x[max_index] + 1:best_end_x[max_index]
    ]

    cv2.namedWindow("img_copy", cv2.WINDOW_NORMAL)
    cv2.imshow("img_copy", img_copy)
    cv2.waitKey(100)
    
    return cropped
