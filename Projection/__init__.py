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
    # x1, mappingX1 = __cubeProjectionX1(panoImage, projectScale)
    # x_1, mappingX_1 = __cubeProjectionX_1(panoImage, projectScale)
    # y1, mappingY1 = __cubeProjectionY1(panoImage, projectScale)
    # y_1, mappingY_1 = __cubeProjectionY_1(panoImage, projectScale)
    # z1, mappingZ1 = __cubeProjectionZ1(panoImage, projectScale)
    # z_1, mappingZ_1 = __cubeProjectionZ_1(panoImage, projectScale)

    x1, mappingX1 = __Projection(panoImage, projectScale, lambda u, v: np.array([1, v, u]))
    x_1, mappingX_1 = __Projection(panoImage, projectScale, lambda u, v: np.array([-1, v, u]))
    y1, mappingY1 = __Projection(panoImage, projectScale, lambda u, v: np.array([v, 1, u]))
    y_1, mappingY_1 = __Projection(panoImage, projectScale, lambda u, v: np.array([v, -1, u]))
    z1, mappingZ1 = __Projection(panoImage, projectScale, lambda u, v: np.array([v, u, 1]))
    z_1, mappingZ_1 = __Projection(panoImage, projectScale, lambda u, v: np.array([v, u, -1]))
    return [
        [x1, mappingX1],
        [x_1, mappingX_1],
        [y1, mappingY1],
        [y_1, mappingY_1],
        [z1, mappingZ1],
        [z_1, mappingZ_1],
    ]


def __Projection(panoImage, projectScale, genRay):
    panoImageShape = panoImage.shape
    projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    # 采样光线
    projectU = 0  # projectU -> projectY
    for du in np.linspace(1, -1, projectScale):  # 列从上到下
        projectV = 0  # projectV -> projectX
        for dv in np.linspace(-1, 1, projectScale):  # 第一行
            ray = genRay(du, dv)
            ray = ray / np.linalg.norm(ray)
            u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
            panoU = round(u * panoImageShape[0])
            panoV = round(v * panoImageShape[1])
            rgb = panoImage[panoU][panoV]
            projectImage[projectU][projectV] = rgb
            projectV += 1
        projectU += 1

    def mapping(x, y):  # y->u,x->v
        # scale 到 [-1,1]
        u = (1 - y / projectScale) * 2 - 1
        v = x / projectScale * 2 - 1
        ray = genRay(u, v)
        ray = ray / np.linalg.norm(ray)
        u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
        my = round(u * panoImageShape[0])
        mx = round(v * panoImageShape[1])
        return mx, my

    return projectImage, mapping


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


def __cubeProjectionX_1(panoImage, projectScale):
    panoImageShape = panoImage.shape
    projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    # 采样光线 y，z，x=-1
    projectU = 0  # projectU -> projectY
    for z in np.linspace(1, -1, projectScale):  # 列从上到下（z）
        projectV = 0  # projectV -> projectX
        for y in np.linspace(-1, 1, projectScale):  # 第一行（y）
            ray = np.array([-1, y, z])
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
        ray = np.array([-1, x, y])
        ray = ray / np.linalg.norm(ray)
        u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
        my = round(u * panoImageShape[0])
        mx = round(v * panoImageShape[1])
        return mx, my

    return projectImage, mapping


def __cubeProjectionY1(panoImage, projectScale):
    panoImageShape = panoImage.shape
    projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    # 采样光线 x，z，y=1
    projectU = 0  # projectU -> projectY
    for z in np.linspace(1, -1, projectScale):  # 列从上到下（z）
        projectV = 0  # projectV -> projectX
        for x in np.linspace(-1, 1, projectScale):  # 第一行（x）
            ray = np.array([x, 1, z])
            ray = ray / np.linalg.norm(ray)
            u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
            panoU = round(u * panoImageShape[0])
            panoV = round(v * panoImageShape[1])
            rgb = panoImage[panoU][panoV]
            projectImage[projectU][projectV] = rgb
            projectV += 1
        projectU += 1

    def mapping(x, y):  # x->x,y->z
        # scale 到 [-1,1]
        x = x / projectScale * 2 - 1
        y = (1 - y / projectScale) * 2 - 1
        ray = np.array([x, 1, y])
        ray = ray / np.linalg.norm(ray)
        u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
        my = round(u * panoImageShape[0])
        mx = round(v * panoImageShape[1])
        return mx, my

    return projectImage, mapping


def __cubeProjectionY_1(panoImage, projectScale):
    panoImageShape = panoImage.shape
    projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    # 采样光线 x，z，y=-1
    projectU = 0  # projectU -> projectY
    for z in np.linspace(1, -1, projectScale):  # 列从上到下（z）
        projectV = 0  # projectV -> projectX
        for x in np.linspace(-1, 1, projectScale):  # 第一行（x）
            ray = np.array([x, -1, z])
            ray = ray / np.linalg.norm(ray)
            u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
            panoU = round(u * panoImageShape[0])
            panoV = round(v * panoImageShape[1])
            rgb = panoImage[panoU][panoV]
            projectImage[projectU][projectV] = rgb
            projectV += 1
        projectU += 1

    def mapping(x, y):  # x->x,y->z
        # scale 到 [-1,1]
        x = x / projectScale * 2 - 1
        y = (1 - y / projectScale) * 2 - 1
        ray = np.array([x, -1, y])
        ray = ray / np.linalg.norm(ray)
        u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
        my = round(u * panoImageShape[0])
        mx = round(v * panoImageShape[1])
        return mx, my

    return projectImage, mapping


def __cubeProjectionZ1(panoImage, projectScale):
    panoImageShape = panoImage.shape
    projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    # 采样光线 x，y，z=1
    projectU = 0  # projectU -> projectY
    for y in np.linspace(1, -1, projectScale):  # 列从上到下（y）
        projectV = 0  # projectV -> projectX
        for x in np.linspace(-1, 1, projectScale):  # 第一行（x）
            ray = np.array([x, y, 1])
            ray = ray / np.linalg.norm(ray)
            u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
            panoU = round(u * panoImageShape[0])
            panoV = round(v * panoImageShape[1])
            rgb = panoImage[panoU][panoV]
            projectImage[projectU][projectV] = rgb
            projectV += 1
        projectU += 1

    def mapping(x, y):  # x->x,y->y
        # scale 到 [-1,1]
        x = x / projectScale * 2 - 1
        y = (1 - y / projectScale) * 2 - 1
        ray = np.array([x, y, 1])
        ray = ray / np.linalg.norm(ray)
        u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
        my = round(u * panoImageShape[0])
        mx = round(v * panoImageShape[1])
        return mx, my

    return projectImage, mapping


def __cubeProjectionZ_1(panoImage, projectScale):
    panoImageShape = panoImage.shape
    projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
    # 采样光线 x，y，z=-1
    projectU = 0  # projectU -> projectY
    for y in np.linspace(1, -1, projectScale):  # 列从上到下（y）
        projectV = 0  # projectV -> projectX
        for x in np.linspace(-1, 1, projectScale):  # 第一行（x）
            ray = np.array([x, y, -1])
            ray = ray / np.linalg.norm(ray)
            u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
            panoU = round(u * panoImageShape[0])
            panoV = round(v * panoImageShape[1])
            rgb = panoImage[panoU][panoV]
            projectImage[projectU][projectV] = rgb
            projectV += 1
        projectU += 1

    def mapping(x, y):  # x->x,y->y
        # scale 到 [-1,1]
        x = x / projectScale * 2 - 1
        y = (1 - y / projectScale) * 2 - 1
        ray = np.array([x, y, -1])
        ray = ray / np.linalg.norm(ray)
        u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
        my = round(u * panoImageShape[0])
        mx = round(v * panoImageShape[1])
        return mx, my

    return projectImage, mapping
