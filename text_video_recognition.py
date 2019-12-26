from imutils.object_detection import non_max_suppression
from imutils.video import VideoStream, FPS
import numpy as np
import pytesseract
import argparse
import imutils
import time
import cv2 as cv

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help='path to input image')
ap.add_argument("-w", '--width', type=int, default=320, help="nearest multiple of 32 for resized width")
ap.add_argument('-e', '--height', type=int, default=320, help='nearest multiple of 32 for resized height')
ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())


def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rects, confidences


(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

print("[INFO] loading EAST text detector...")
net = cv.dnn.readNet("frozen_east_text_detection.pb")

print("[INFO] starting video stream...")
vs = cv.VideoCapture(0)
time.sleep(1.0)

fps = FPS().start()

while True:
    ret, frame = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=1000)
    orig = frame.copy()

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        rW = W / float(newW)
        rH = H / float(newH)

    frame = cv.resize(frame, (newW, newH))
    blob = cv.dnn.blobFromImage(frame, 1.0, (newW, newH),
                                (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        roi = orig[startY:endY, startX:endX]
        config = "-l eng --oem 1 --psm 7"
        text = pytesseract.image_to_string(roi, config=config)

        print("OCR TEXT: {}\n".format(text))

        cv.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv.putText(orig, text, (startX, startY - 20), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    fps.update()

    cv.imshow("Text Detection", orig)
    cv.resizeWindow("Text Detection", 2000, 2000)
    key = cv.waitKey(1) & 0xFF

    if key == ord('q'):
        break


fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv.destroyAllWindows()
