import cv2
import numpy as np
import Projection
import VpEstimation

panoImage = cv2.imread("../images/360_1.jpg", cv2.IMREAD_COLOR)
panoImage = cv2.imread("../images/vp.png", cv2.IMREAD_COLOR)
print(panoImage.shape)

projectScale = 500

vp = []

for v in np.linspace(0.1, 0.9, 9):
    for u in np.linspace(0.1, 0.9, 9):
        print(u, v)
        projectImage, mapping = Projection.RayProjection(panoImage, projectScale, u, v)
        ret, x, y = VpEstimation.HoughVpEstimation(projectImage)
        X, Y = mapping(x, y)
        vp.append([X, Y])
        cv2.circle(panoImage, (X, Y), 20, (0, 255, 0), -1)
        # ret.append([projectImage, mapping])

cv2.imwrite("output_pano_vps.jpg", panoImage)

print(vp)
# projectImageAndMapping = Projection.__demoRayProjection(panoImage, 500)
#
# panoEdgesImage = cv2.imread('./output_cube_project_1.jpg')
# panoEdgesImage = cv2.imread('./output_ray_project_19.jpg')
# panoEdgesImage = cv2.imread('./output_ray_project_20.jpg')
# VpEstimation.HoughVpEstimation(panoEdgesImage)
# print(x, y)

# cv2.circle(panoEdgesImage, (round(x), round(y)), 20, (0, 0, 0), -1)
# cv2.imshow("vp", panoEdgesImage)
# cv2.waitKey(0)
# cv2.imwrite("output_line_circle_normal.jpg", lineCircleNormal)
