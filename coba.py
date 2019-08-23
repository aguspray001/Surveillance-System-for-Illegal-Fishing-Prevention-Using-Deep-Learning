import sys
sys.path.append('/usr/local/lib/python3.5/dist-packages')

import cv2 as cv
# Load the model.
net = cv.dnn.readNet('kapal.xml',
                     'kapal.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
# Read an image.
frame = cv.imread('/home/aguspray/ship_detection/Video/2.jpg')
if frame is None:
    raise Exception('Image not found!')
# Prepare input blob and perform an inference.
blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U)
net.setInput(blob)
out = net.forward()
# Draw detected faces on the frame.
for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])
    if confidence > 0.3:
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(50, 255, 0),thickness=2)
# Save the frame to an image file.
cv.imwrite('objek_deteksi_openvino.png', frame)
