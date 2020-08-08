import cv2
import VpEstimation

panoImage = cv2.imread("../images/360_1.jpg", cv2.IMREAD_COLOR)
# panoImage = cv2.imread("../dataset-good/dataset-" + str(13) + ".jpg", cv2.IMREAD_COLOR)
print(panoImage.shape)
VpEstimation.HoughGreatCircleVpEstimationSegmentationAndLineIndex(panoImage, "line_index_360_1")

# for i in range(1, 30):
#     panoImage = cv2.imread("../dataset-good/dataset-" + str(i) + ".jpg", cv2.IMREAD_COLOR)
#     print(panoImage.shape)
#     VpEstimation.HoughGreatCircleVpEstimationSegmentation(panoImage, logid="tt5" + str(i))
