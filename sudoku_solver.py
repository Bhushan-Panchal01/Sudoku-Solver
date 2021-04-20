from utils import *

##########################################
pathImage = "resources/img2.jpg"
heightImg = 450
widthImg = 450
model = initialize()
##########################################


# 1. PREPARE IMAGE
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
imgThresh = preProcess(img)

# 2. Finding contours
imgContours = img.copy()
imgBigContours = img.copy()
contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3, lineType=None, hierarchy=None, maxLevel=None, offset=None)


# 3. finding biggesrt contour
biggest, maxArea = biggestContour(contours)
print(biggest)
if biggest.size != 0:
	biggest = reorder(biggest)
	cv2.drawContours(imgBigContours, biggest, -1, (0, 255, 0), thickness=10, lineType=None, hierarchy=None, maxLevel=None, offset=None)

	pts1 = np.float32(biggest)
	pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg), dst=None, flags=None, borderMode=None, borderValue=None)
	imgDetectedDigits = imgBlank.copy()
	imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY, dst=None, dstCn=None)

# 4. Image Splitting into single block 
imgSolvedDigits = imgBlank.copy()
boxes = splitBoxes(imgWarpColored)
numbers = getPredection(boxes, model)



# Image show
cv2.imshow('Image', imgWarpColored)
cv2.waitKey(delay=0)