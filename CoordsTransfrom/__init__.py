"""
xyz:          [-1,1], [-1,1], [-1,1], 单位球面, 右手坐标系, z 轴向上
u, v:         [0,1], [0,1], u 向下, v 向右
phi, theta:   [0,pi], [0,2pi], phi 与 z 轴夹角, theta 与 x 轴夹角
x, y:         [0,scaleW（shape[1]）], [0,scaleH（shape[0]）], 像素坐标，x->v，y->u
"""

import math


def phi_theta2xyz(phi, theta):
    z = math.cos(phi)
    rP = math.sin(phi)  # 投影半径
    x = math.cos(theta) * rP
    y = math.sin(theta) * rP
    return x, y, z


def xyz2phi_theta(x, y, z):
    phi = math.acos(z)
    theta = math.atan2(y, x)
    if theta < 0:
        theta += math.pi * 2
    return phi, theta


def phi_theta2uv(phi, theta):
    u = phi / math.pi
    v = theta / (2 * math.pi)
    return u, v


def uv2phi_theta(u, v):
    phi = u * math.pi
    theta = v * 2 * math.pi
    return phi, theta


# 若u，v为1，scale后访问像素数组会越界；xyz2uv(0, 0, -1)->(1.0, 0.0)
def xyz2uv(x, y, z):
    phi, theta = xyz2phi_theta(x, y, z)
    u, v = phi_theta2uv(phi, theta)
    return u, v


def uv2xyz(u, v):
    phi, theta = uv2phi_theta(u, v)
    x, y, z = phi_theta2xyz(phi, theta)
    return x, y, z


def sphere2fakecube(x, y, z):
    xyzs = []
    if x > 0.5:
        xyzs.append([1, y, z])
    else:
        xyzs.append([-1, y, z])
    if y > 0.5:
        xyzs.append([x, 1, z])
    else:
        xyzs.append([x, -1, z])
    if z > 0.5:
        xyzs.append([x, y, 1])
    else:
        xyzs.append([x, y, -1])
    return xyzs


def sphere2cube(x, y, z):
    maxValue = max(abs(x), abs(y), abs(z))
    scale = 1 / maxValue
    return [[x * scale, y * scale, z * scale]]


def uv2xy(u, v, scaleShape):
    x = round(v * scaleShape[1])
    y = round(u * scaleShape[0])
    # 防止像素数组越界
    x = min(x, scaleShape[1] - 1)
    y = min(y, scaleShape[0] - 1)
    return x, y


def xy2uv(x, y, scaleShape):
    u = y / scaleShape[0]
    v = x / scaleShape[1]
    return u, v


def xy2xyz(x, y, scaleShape):
    u, v = xy2uv(x, y, scaleShape)
    return uv2xyz(u, v)


def xyz2xy(x, y, z, scaleShape):
    u, v = xyz2uv(x, y, z)
    return uv2xy(u, v, scaleShape)


def xy2phi_theta(x, y, scaleShape):
    u, v = xy2uv(x, y, scaleShape)
    phi, theta = uv2phi_theta(u, v)
    return phi, theta


def phi_theta2xy(phi, theta, scaleShape):
    u, v = phi_theta2uv(phi, theta)
    x, y = uv2xy(u, v, scaleShape)
    return x, y


########## test ##########

if __name__ == "__main__":

    N = 1000
    epsilon = 0.0000000001

    # phi 为 0 或 pi，theta 为 2pi 时无法等价转换
    for i in range(1, N):
        for j in range(0, N):
            phi = math.pi / N * i
            theta = math.pi * 2 / N * j
            x, y, z = phi_theta2xyz(phi, theta)
            phiN, thetaN = xyz2phi_theta(x, y, z)
            if math.fabs(phi - phiN) > epsilon:
                print("error(phi):", phi, phiN)
            if math.fabs(theta - thetaN) > epsilon:
                print("error(theta):", theta, thetaN)

    print(xyz2phi_theta(0, 0, 1))
    print(xyz2phi_theta(0, 0, -1))
    print(xyz2uv(0, 0, -1))
