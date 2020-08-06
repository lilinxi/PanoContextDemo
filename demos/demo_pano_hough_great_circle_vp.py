import cv2
import numpy as np
import Projection
import VpEstimation

panoImage = cv2.imread("../images/360.jpg", cv2.IMREAD_COLOR)
print(panoImage.shape)

projectScale = 500

projectImageAndMappings = Projection.CubeProjection(panoImage, projectScale)

VpEstimation.HoughGreatCircleVpEstimation(panoImage, projectImageAndMappings)

# cv2.imwrite("output_hough_great_circle_normal_p.jpg", panoImageWithGreatCircleNormal)
