"""
    This python code is for detecting facial blobs given a video

        Input: a video containing speakers
        Output: detected facial blobs in highlighted rectangles frame by frame
"""

import os
import cv2
import numpy as np

video_path = r"..." # absolute path for the video

# Define paths for pre-requisite models
base_dir = os.path.dirname(__file__)
prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel')

# Read the pre-trained facial blobs detection model from OpenCV
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

vc = cv2.VideoCapture(video_path)
ret, first_frame = vc.read()
resize_dim = 600

while (vc.isOpened()):

    max_dim = max(first_frame.shape)
    scale = resize_dim / max_dim
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)

    (h, w) = first_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(first_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    # Create frame around face
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        if(startX < 0):
            startX = 0
        if(startY < 0):
            startY = 0
        if(endX < 0):
            endX = 0
        if(endY < 0):
            endY = 0

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            cv2.rectangle(first_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            break
        else:
            break

    # Display
    cv2.imshow('facial blobs', first_frame)

    # Skip internal frames, number = fps
    fps = vc.get(cv2.CAP_PROP_FPS)
    for j in range(int(fps) - 1):
        vc.grab()

    # Read the frame on the next second
    ret, next_frame = vc.read()
    first_frame = next_frame

    # The programs breaks out of the while loop when the user presses the ‘q’ key
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()