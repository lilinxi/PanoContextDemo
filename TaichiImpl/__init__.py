import taichi as ti
import numpy as np
import cv2

ti.init(arch=ti.cpu)

PanoImageShape = (1000, 2000)
PanoImage = ti.Vector.field(3, dtype=ti.uint8, shape=PanoImageShape)


def ReadImage(filename):
    panoImage = cv2.imread(filename, cv2.IMREAD_COLOR)
    panoImage = cv2.resize(panoImage, (2000, 1000), interpolation=cv2.INTER_AREA)
    PanoImage.from_numpy(panoImage)


def WriteImage(filename):
    panoImage = PanoImage.to_numpy()
    cv2.imwrite(filename, panoImage)


# def CubeProjection(panoImage, projectScale):
#     """
#     全景图转为六面体的投影图
#     六面体：x,y,z -> [-1,1], [-1,1], [-1,1]
#     :param panoImage:       全景图（2X1 视角，彩色）cv2.imread('', cv2.IMREAD_COLOR)
#     :param projectScale:    投影图大小
#     :return:                投影图，投影图到全景图的映射（带scale的映射）
#     """
#     # x1, mappingX1 = __cubeProjectionX1(panoImage, projectScale)
#     # x_1, mappingX_1 = __cubeProjectionX_1(panoImage, projectScale)
#     # y1, mappingY1 = __cubeProjectionY1(panoImage, projectScale)
#     # y_1, mappingY_1 = __cubeProjectionY_1(panoImage, projectScale)
#     # z1, mappingZ1 = __cubeProjectionZ1(panoImage, projectScale)
#     # z_1, mappingZ_1 = __cubeProjectionZ_1(panoImage, projectScale)
#
#     x1, mappingX1 = __Projection(panoImage, projectScale, lambda u, v: np.array([1, v, u]))
#     print("x1")
#     x_1, mappingX_1 = __Projection(panoImage, projectScale, lambda u, v: np.array([-1, v, u]))
#     print("x_1")
#     y1, mappingY1 = __Projection(panoImage, projectScale, lambda u, v: np.array([v, 1, u]))
#     print("y1")
#     y_1, mappingY_1 = __Projection(panoImage, projectScale, lambda u, v: np.array([v, -1, u]))
#     print("y_1")
#     z1, mappingZ1 = __Projection(panoImage, projectScale, lambda u, v: np.array([v, u, 1]))
#     print("z1")
#     z_1, mappingZ_1 = __Projection(panoImage, projectScale, lambda u, v: np.array([v, u, -1]))
#     print("z_1")
#     return [
#         [x1, mappingX1],
#         [x_1, mappingX_1],
#         [y1, mappingY1],
#         [y_1, mappingY_1],
#         [z1, mappingZ1],
#         [z_1, mappingZ_1],
#     ]
#
#
# def __Projection(panoImage, projectScale, genRay):
#     panoImageShape = panoImage.shape
#     projectImage = np.zeros((projectScale, projectScale, 3), dtype=np.uint8)
#     # 采样光线
#     projectU = 0  # projectU -> projectY
#     for du in np.linspace(1, -1, projectScale):  # 列从上到下
#         projectV = 0  # projectV -> projectX
#         for dv in np.linspace(-1, 1, projectScale):  # 第一行
#             ray = genRay(du, dv)
#             ray = ray / np.linalg.norm(ray)
#             u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
#             panoU = int(u * panoImageShape[0])  # TODO 用 int 而不用 round，用 round 在 u，v 等于零和接近1的时候会丢失半个像素的信息
#             panoV = int(v * panoImageShape[1])
#             panoU = min(panoImageShape[0] - 1, panoU)
#             panoV = min(panoImageShape[1] - 1, panoV)
#             rgb = panoImage[panoU][panoV]
#             projectImage[projectU][projectV] = rgb
#             projectV += 1
#         projectU += 1
#
#     def mapping(x, y):  # y->u,x->v
#         # scale 到 [-1,1]
#         u = (1 - y / projectScale) * 2 - 1
#         v = x / projectScale * 2 - 1
#         ray = genRay(u, v)
#         ray = ray / np.linalg.norm(ray)
#         u, v = CoordsTransfrom.xyz2uv(ray[0], ray[1], ray[2])
#         my = round(u * panoImageShape[0])
#         mx = round(v * panoImageShape[1])
#         return mx, my
#
#     return projectImage, mapping

@ti.kernel
def test_taichi():
    for x, y in PanoImage:
        PanoImage[x, y][1] = 0


def test_taichi_p():
    for x in range(1000):
        for y in range(2000):
            PanoImage[x, y][1] = 0


@ti.func
def GenRay(a: ti.Vector):
    return ti.Vector([1, a[0], a[1]])


@ti.kernel
def ProjectionTaichi():
    a = ti.Vector([2, 3])
    print(a)
    # GenRay = lambda a: ti.Vector([1, a[0], a[1]])
    # print(GenRay(a))
    pass


def t():
    ReadImage("../images/360_1.jpg")
    test_taichi()
    WriteImage("test.jpg")


@ti.kernel
def TestP():
    print("be")
    for i in range(100):
        print(i)


@ti.kernel
def inside_taichi_scope():
    x = 233
    print('hello', x)


if __name__ == "__main__":
    # TestP()
    # inside_taichi_scope()
    # t()
    # ProjectionTaichi()
    ReadImage("../images/360_1.jpg")
    test_taichi()
    WriteImage("./output_test_p.jpg")
