import cv2
import numpy as np

import Projection
import Visualization

panoImage = cv2.imread("../images/360_1.jpg", cv2.IMREAD_COLOR)
print(type(panoImage))
print(panoImage.shape)

projectImage, mapping = Projection.__cubeProjectionX1(panoImage, 500)
print(type(projectImage))
print(projectImage.shape)

cv2.imwrite("output_cube_project_x1.jpg", projectImage)

projectImageGray = cv2.cvtColor(projectImage, cv2.COLOR_BGR2GRAY)
fld = cv2.ximgproc.createFastLineDetector()
lines = fld.detect(projectImageGray)

Visualization.DrawPanoLine(panoImage, lines, mapping, (255, 0, 0))
# Visualization.__drawPanoLineTest(panoImage, lines, mapping, (255, 0, 0))

cv2.imwrite("output_pano_x1_lsd.jpg", panoImage)

edges = np.full(panoImage.shape,255, dtype=np.uint8)
Visualization.DrawPanoLine(edges, lines, mapping, (255, 0, 0))

cv2.imwrite("output_pano_x1_edges.jpg", edges)
