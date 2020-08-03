import cv2
import numpy as np

"""
给 python 打开窗口的权限

```
defaults write org.python.python ApplePersistenceIgnoreState NO
```
"""

"""
cv2.error: OpenCV(4.3.0) /Users/travis/build/skvark/opencv-python/opencv/modules/imgproc/src/lsd.cpp:143: error: (-213:The function/feature is not implemented) Implementation has been removed due original code license issues in function 'LineSegmentDetectorImpl'
"""

# img = cv2.imread('./images/360_1.jpg', cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# lsd = cv2.createLineSegmentDetector(0, _scale=1)
#
# dlines = lsd.detect(gray)
#
# for dline in dlines[0]:
#     x0 = int(round(dline[0][0]))
#     y0 = int(round(dline[0][1]))
#     x1 = int(round(dline[0][2]))
#     y1 = int(round(dline[0][3]))
#     cv2.line(img, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
#
# cv2.imshow("lsd", img)


img = cv2.imread('/Users/limengfan/PycharmProjects/PanoContext/images/360_1.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fld = cv2.ximgproc.createFastLineDetector()
lines = fld.detect(gray)

print(type(lines))  # <class 'numpy.ndarray'>: numpy.ndarray中的每个元素的dtype应该为numpy.uint8
print(np.array(lines).shape)
for i in range(0, 10):
    print(lines[i])

result_img = fld.drawSegments(img, lines)
cv2.imwrite("output_lsd.jpg", result_img)

