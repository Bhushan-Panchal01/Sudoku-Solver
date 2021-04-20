import cv2
import  numpy as np

# 1.. Preprocessing image
def preProcess(img):
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1, dst=None, sigmaY=None, borderType=None)
	imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2, dst=None)
	return imgThreshold


# 2. finding biggest contour
def biggestContour(contours):
	biggest = np.array([])
	max_area = 0
	for i in contours:
		area = cv2.contourArea(i, oriented=None)
		if area > 50:
			peri = cv2.arcLength(i, True)
			approx = cv2.approxPolyDP(i, 0.02 * peri, True)
			if area > max_area and len(approx) == 4:
				biggest = approx
				max_area = area


	return biggest, max_area

# 3. reorder points for wrap perspective
def reorder(myPoints):
	myPoints = myPoints.reshape((4,2))
	myPointsNew = np.zeros((4,1,2), dtype=np.int32)

	add = myPoints.sum(1)
	myPointsNew[0] = myPoints[np.argmin(add)]
	myPointsNew[3] = myPoints[np.argmax(add)]

	diff = np.diff(myPoints, axis=1)
	myPointsNew[1] = myPoints[np.argmin(diff)]
	myPointsNew[2] = myPoints[np.argmax(diff)]

	return myPointsNew

# 4. Splitting each digit inito single image
def splitBoxes(img):
	rows = np.vsplit(img, 9)
	boxes = [] 
	for r in rows:
		cols = np.hsplit(r, 9)
		for box in cols:
			boxes.append(box)
	return boxes