import cv2
import VpEstimation

panoImage = cv2.imread("../images/360_1.jpg", cv2.IMREAD_COLOR)
print(panoImage.shape)

VpEstimation.HoughGreatCircleVpEstimationTest(panoImage)
