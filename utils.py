import cv2
import numpy as np
import pickle
import time

# Model initialization
def initialize():
    pickle_in = open('model_test.p', 'rb')
    model = pickle.load(pickle_in)
    return model


# 1.. Preprocessing image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold


# 2. finding biggest contour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print(approx)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


# 3. reorder points for wrap perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)

    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


# 4. Splitting each digit into single image
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


# 4. Get Predictions for those images
def getPrediction(boxes, model):
    result = []
    # Preprocessing
    for image in boxes:
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4: img.shape[1] - 4]  # removing extra details in the image
        img = cv2.resize(img, (32, 32))
        img = img / 255
        # print(img.shape)
        img = img.reshape(1, 32, 32, 1)

        # prediction
        preds = model.predict(img)

        classIndex = np.argmax(preds, axis=-1)  # predicts class
        probVal = np.amax(preds)  # prediction accuracy

        # print(classIndex, probVal)

        # Approval at threshold
        if probVal > 0.9:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


def displayNumbers(img, numbers, color):
    secW = int(img.shape[0] / 9)
    secH = int(img.shape[1] / 9)

    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y * 9) + x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]),
                            (x*secW+int(secW/2)-10, int((y+0.8)*secH)),
                             cv2.FONT_HERSHEY_PLAIN, 3, color, 2, cv2.LINE_AA)
    return img