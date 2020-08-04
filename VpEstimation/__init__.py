import numpy as np
import cv2
import CoordsTransfrom
import time


def __sampleLineCircleNormal(point, sampleN=360):
    """
    采样一个点所在的sampleN个大圆的法线方向
    :param point:       point 可以作为从圆心开始的一个方向（已归一化）
    :param sampleN:     采样的法线数目
    :return:            采样的法线列表
    """
    # TODO step1. 建立局部坐标系
    if abs(point[0]) > abs(point[1]):  # x 必不为零
        v1 = np.array([-point[2], 0, point[0]])
    else:  # x=y=0, z 必不为零；y>x, y 必不为零
        v1 = np.array([0, point[2], -point[1]])
    v2 = np.cross(point, v1)
    # TODO step2. [0,2pi) 采样
    samples = np.zeros((sampleN, 3))
    n = 0
    for theta in np.linspace(0, 2 * np.pi, sampleN, endpoint=False):
        samples[0] = np.cos(theta) * v1 + np.sin(theta) * v2
    return samples


def __drawLineCircleNormal(lineCircleNormal, panoEdgesGray, bgr=(0, 255, 0), panoEdgesGrayZero=255):
    """
    3D空间中的线段对应于全景球体上大圆的一部分，并在全景图像中显示为曲线。
    对于每条直线l，我们使用n表示其大圆所在的平面的法线方向。
    与直线l相关联的消失方向v应该垂直于n。
    我们使用霍夫变换查找所有消失的方向。
    :param lineCircleNormal:    绘制的图
    :param panoEdges:           灰度图，对所有不为255的值进行霍夫检测
    """
    panoEdgesShape = panoEdgesGray.shape
    for y in range(panoEdgesShape[0]):
        if y % 10 == 0:
            print(y)
        for x in range(panoEdgesShape[1]):
            if panoEdgesGray[y][x] != panoEdgesGrayZero:
                u, v = y / panoEdgesShape[0], x / panoEdgesShape[1]
                dx, dy, dz = CoordsTransfrom.uv2xyz(u, v)
                samples = __sampleLineCircleNormal(np.array([dx, dy, dz]))
                for sample in samples:
                    lu, lv = CoordsTransfrom.xyz2uv(sample[0], sample[1], sample[2])
                    lx, ly = CoordsTransfrom.uv2xy(lu, lv, lineCircleNormal.shape)
                    cv2.circle(lineCircleNormal, (lx, ly), 1, bgr, -1)
