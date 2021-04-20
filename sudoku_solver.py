from utils import *
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3
from sudokuLogic import solve

##########################################
heightImg = 450
widthImg = 450
model = initialize()
##########################################


def sudoku_solver(pathImage):
	# 1. PREPARE IMAGE
	img = cv2.imread(pathImage)
	img = cv2.resize(img, (widthImg, heightImg))
	imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
	imgThresh = preProcess(img)

	# 2. Finding contours
	imgContours = img.copy()
	imgBigContours = img.copy()
	contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

	# 3. finding biggest contour
	biggest, maxArea = biggestContour(contours)
	if biggest.size != 0:
		biggest = reorder(biggest)
		cv2.drawContours(imgBigContours, biggest, -1, (0, 255, 0), thickness=10)

		pts1 = np.float32(biggest)
		pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

		matrix = cv2.getPerspectiveTransform(pts1, pts2)

		imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

		imgDetectedDigits = imgBlank.copy()
		imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

		# 4. Image Splitting into single block
		imgSolvedDigits = imgBlank.copy()
		boxes = splitBoxes(imgWarpColored)

		numbers = getPrediction(boxes, model)

		# imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color = (255,100,0))
		# print(numbers)

		numbers = np.asarray(numbers)
		posArray = np.where(numbers>0, 0, 1)
		# print(posArray)

		# 5. Find Solution
		board = np.array(numbers)
		board = board.reshape(9,9)

		solve(board)

		flatList = []
		for sublist in board:
			for item in sublist:
				flatList.append(item)
		solvedNumbers = flatList*posArray
		imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers, color=(0, 255, 0))

		# 6. Overlay
		pts2 = np.float32(biggest)
		pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

		matrix = cv2.getPerspectiveTransform(pts1, pts2)
		imgWarpColored2 = img.copy()
		imgWarpColored2 = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
		inv_perspective = cv2.addWeighted(imgWarpColored2, 1, img, .4, 1)

		# Image show
		cv2.imshow('Solution', inv_perspective)
		cv2.waitKey(0)