import Segmentation
import CoordsTransfrom
import cv2
import numpy as np

shape = (1000, 2000, 3)

demoSegmentation = np.zeros(shape, dtype=np.uint8)

phi_theta2segmentation, _ = Segmentation.SolidAngleSegmentation((5, 10))

for y in range(shape[0]):
    for x in range(shape[1]):
        phi, theta = CoordsTransfrom.xy2phi_theta(x, y, shape)
        i, j = phi_theta2segmentation(phi, theta)
        demoSegmentation[y][x] = (i * 50, 100, j * 20)

cv2.imwrite("output_demo_segmentation.jpg", demoSegmentation)
