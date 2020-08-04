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
    六面体：x,y,z -> [-1,1], [-1,1], [-1,1]
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
            panoU = int(u * panoImageShape[0])  # TODO 用 int 而不用 round，用 round 在 u，v 等于零和接近1的时候会丢失半个像素的信息
            panoV = int(v * panoImageShape[1])
            panoU = min(panoImageShape[0] - 1, panoU)
            panoV = min(panoImageShape[1] - 1, panoV)
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


def BuildCoords(normal):
    """
    建立局部坐标系(右手坐标系)
    :param normal:  已经正则化，作为坐标系的x轴
    :return:        坐标系的y轴，z轴
    """
    if normal[1] != 0:
        v1 = np.array([-normal[1], normal[0], 0])
    else:
        v1 = np.array([0, 1, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    return v1, v2
    # if abs(normal[0]) > abs(normal[1]):  # x 必不为零
    #     v1 = np.array([-normal[2], 0, normal[0]])
    # else:  # x=y=0, z 必不为零；y>x, y 必不为零
    #     v1 = np.array([0, normal[2], -normal[1]])
    # v2 = np.cross(v1, normal)
    # return v1, v2


def RayProjection(panoImage, projectScale, u, v):
    x, y, z = CoordsTransfrom.uv2xyz(u, v)
    normal = np.array([x, y, z])
    v1, v2 = BuildCoords(normal)
    rotateMat = np.array([normal, v1, v2]).T  # 将 ray(x,y,z) 旋转到 (1,0,0) 的旋转矩阵
    # for du in np.linspace(1, -1, projectScale):  # 列从上到下
    # for dv in np.linspace(-1, 1, projectScale):  # 第一行
    genRay = lambda u, v: rotateMat.dot(np.array([1, v, u]))
    return __Projection(panoImage, projectScale, genRay)


def __demoRayProjection(panoImage, projectScale):
    ret = []
    for v in np.linspace(0.3, 0.7, 5):
        for u in np.linspace(0.3, 0.7, 5):
            print(u, v)
            projectImage, mapping = RayProjection(panoImage, projectScale, u, v)
            ret.append([projectImage, mapping])
    # u = 0.5
    # v = 0.5
    # projectImage, mapping = RayProjection(panoImage, projectScale, u, v)
    # ret.append([projectImage, mapping])

    return ret
