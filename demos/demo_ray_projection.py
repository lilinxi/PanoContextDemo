import cv2
import Projection

panoImage = cv2.imread("../images/360_1.jpg", cv2.IMREAD_COLOR)

projectImageAndMapping = Projection.__demoRayProjection(panoImage, 500)

for i in range(len(projectImageAndMapping)):
    cv2.imwrite("output_ray_project_" + str(i) + ".jpg", projectImageAndMapping[i][0])
