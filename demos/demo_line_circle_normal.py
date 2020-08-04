import cv2
import numpy as np
import VpEstimation

panoEdgesImage = cv2.imread('./output_pano_edges.jpg')
panoEdgesGray = cv2.cvtColor(panoEdgesImage, cv2.COLOR_BGR2GRAY)
print(panoEdgesGray.shape)
# print(np.count_nonzero(panoEdgesGray == 255))
# print(np.count_nonzero(panoEdgesGray != 255))

lineCircleNormal = np.zeros(shape=panoEdgesImage.shape, dtype=np.uint8)
VpEstimation.__drawLineCircleNormal(lineCircleNormal, panoEdgesGray)
cv2.imwrite("output_line_circle_normal.jpg", lineCircleNormal)

