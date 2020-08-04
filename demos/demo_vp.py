import cv2
import numpy as np
import VpEstimation

panoEdgesImage = cv2.imread('./output_cube_project_1.jpg')
x, y = VpEstimation.RansacVpEstimation(panoEdgesImage)
print(x, y)

cv2.circle(panoEdgesImage, (round(x), round(y)), 20, (0, 0, 0), -1)
cv2.imshow("vp", panoEdgesImage)
cv2.waitKey(0)
# cv2.imwrite("output_line_circle_normal.jpg", lineCircleNormal)
