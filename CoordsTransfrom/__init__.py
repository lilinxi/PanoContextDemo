"""
xyz:          单位球面, 右手坐标系, z 轴向上
u, v:         [0,1], [0,1], u 向下, v 向右
phi, theta:   [0,pi], [0,2pi], phi 与 z 轴夹角, theta 与 x 轴夹角
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


def xyz2uv(x, y, z):
    phi, theta = xyz2phi_theta(x, y, z)
    u, v = phi_theta2uv(phi, theta)
    return u, v


def uv2xyz(u, v):
    phi, theta = uv2phi_theta(u, v)
    x, y, z = phi_theta2xyz(phi, theta)
    return x, z, y


def sphere2cube(x, y, z):
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


########## test ##########

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
