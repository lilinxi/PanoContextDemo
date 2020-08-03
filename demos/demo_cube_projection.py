import cv2
import Projection

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

for line in lines:
    x0 = int(round(line[0][0]))
    y0 = int(round(line[0][1]))
    x1 = int(round(line[0][2]))
    y1 = int(round(line[0][3]))
    cv2.line(projectImage, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)

cv2.circle(projectImage, (0,0), 5, (255, 0, 0))

cv2.imwrite("output_cube_project_x1_lsd.jpg", projectImage)
