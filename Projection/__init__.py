import CoordsTransfrom

import numpy as np


def DemoProjection(panoImage, projectScale):
    projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)

    def mapping(x, y):
        return x, y

    return [
        [projectImage, mapping]
    ]


def CubeProjection(panoImage, projectScale):
    """
    全景图转为六面体的投影图
    :param panoImage:       全景图（2X1 视角，彩色）cv2.imread('', cv2.IMREAD_COLOR)
    :param projectScale:    投影图大小
    :return:                投影图，投影图到全景图的映射（带scale的映射）
    """
    x1 = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    x_1 = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    y1 = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    y_1 = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    z1 = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    z_1 = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    for i in range(0, panoImage.shape[0]):
        for j in range(0, panoImage.shape[1]):
            u = i / panoImage.shape[0]
            v = j / panoImage.shape[1]
            rgb = panoImage[i][j]
            x, y, z = CoordsTransfrom.uv2xyz(u, v)
            if abs(x) > abs(y) and abs(x) > abs(z):
                pass
            elif abs(y) > abs(x) and abs(y) > abs(z):
                pass
            elif abs(z) > abs(x) and abs(z) > abs(x):
                pass
            else:
                pass

            # maxValue = max(abs(x), abs(y), abs(z))
            # projectScale = 1 / maxValue
            # return [[x * projectScale, y * projectScale, z * projectScale]]
            # xyzs = CoordsTransfrom.sphere2cube(x, y, z)
            # points.extend(xyzs)
            # colors.extend([rgb] * len(xyzs))
    pass


def __cubeProjectionX1(panoImage, projectScale):
    panoImageShape = panoImage.shape
    projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    # 采样光线 y，z，x=1
    projectU = 0  # projectU -> projectY
    for z in np.linspace(1, -1, projectScale):  # 列从上到下（z）
        projectV = 0  # projectV -> projectX
        for y in np.linspace(-1, 1, projectScale):  # 第一行（y）
            ray = np.array([1, y, z])
            ray = ray / np.linalg.norm(ray)
            u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
            panoU = round(u * panoImageShape[0])
            panoV = round(v * panoImageShape[1])
            rgb = panoImage[panoU][panoV]
            projectImage[projectU][projectV] = rgb
            projectV += 1
        projectU += 1

    def mapping(x, y):  # x->y,y->z
        # scale 到 [-1,1]
        x = x / projectScale * 2 - 1
        y = (1 - y / projectScale) * 2 - 1
        ray = np.array([1, x, y])
        ray = ray / np.linalg.norm(ray)
        u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
        my = round(u * panoImageShape[0])
        mx = round(v * panoImageShape[1])
        return mx, my

    return projectImage, mapping
