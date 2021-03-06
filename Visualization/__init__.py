import numpy as np
import cv2


def __drawPanoLineTest(panoImage, lines, mapping, bgr, sampleRate=1.1):
    x, y = mapping(100, 100)
    print(x, y)
    cv2.circle(panoImage, (x, y), 300, (0, 0, 0), -1)


def DrawPanoLine(panoImage, lines, mapping, bgr, sampleRate=1.1):
    """
    在全景图中绘制投影直线
    :param panoImage:       全景图
    :param lines:           投影直线
    :param mapping:         投影图到全景图的映射（带scale）
    :param bgr:             颜色
    :param sampleRate:      绘制直线的采样率
    """
    for line in lines:
        x0 = line[0][0]
        y0 = line[0][1]
        x1 = line[0][2]
        y1 = line[0][3]
        dx = x1 - x0
        dy = y1 - y0

        mx0, my0 = mapping(x0, y0)
        mx1, my1 = mapping(x1, y1)
        samples = round(max(abs(mx0 - mx1), abs(my0 - my1)) * sampleRate)

        for dt in np.linspace(0, 1, samples):
            x = x0 + dt * dx
            y = y0 + dt * dy
            mx, my = mapping(x, y)
            cv2.circle(panoImage, (mx, my), 1, bgr, -1)
