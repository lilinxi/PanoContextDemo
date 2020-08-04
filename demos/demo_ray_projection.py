import cv2
import Projection

panoImage = cv2.imread("../images/vp.png", cv2.IMREAD_COLOR)
print(panoImage.shape)

projectImageAndMapping = Projection.__demoRayProjection(panoImage, 500)

for i in range(len(projectImageAndMapping)):
    cv2.imwrite("output_ray_project_" + str(i) + ".jpg", projectImageAndMapping[i][0])
