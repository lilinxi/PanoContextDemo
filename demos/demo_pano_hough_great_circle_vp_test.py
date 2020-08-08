import cv2
import VpEstimation

# panoImage = cv2.imread("../images/360_1.jpg", cv2.IMREAD_COLOR)
# print(panoImage.shape)
#
# VpEstimation.HoughGreatCircleVpEstimationTest(panoImage)

for i in range(17, 32):
    panoImage = cv2.imread("../dataset-good/dataset-" + str(i) + ".jpg", cv2.IMREAD_COLOR)
    print(panoImage.shape)
    VpEstimation.HoughGreatCircleVpEstimationTest(panoImage, logid=i + 100)
