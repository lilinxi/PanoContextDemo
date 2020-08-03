import cv2
import numpy as np

import Projection
import Visualization

panoImage = cv2.imread("../images/360_1.jpg", cv2.IMREAD_COLOR)
edges = np.full(panoImage.shape, 255, dtype=np.uint8)

projectImageAndMapping = Projection.CubeProjection(panoImage, 500)

for i in range(len(projectImageAndMapping)):
    cv2.imwrite("output_cube_project_" + str(i) + ".jpg", projectImageAndMapping[i][0])
    projectImage=projectImageAndMapping[i][0]
    mapping=projectImageAndMapping[i][1]
    projectImageGray = cv2.cvtColor(projectImage, cv2.COLOR_BGR2GRAY)
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(projectImageGray)
    Visualization.DrawPanoLine(panoImage, lines, mapping, (255, 0, 0))
    Visualization.DrawPanoLine(edges, lines, mapping, (255, 0, 0))

cv2.imwrite("output_pano_lsd.jpg", panoImage)
cv2.imwrite("output_pano_edges.jpg", edges)
