import random
import time

import numpy as np
import cv2
import CoordsTransfrom
import Projection
import Visualization


def HoughGreatCircleVpEstimationTest(panoImage, samples=360, logid=0):
    panoImage = cv2.resize(panoImage, (2000, 1000), interpolation=cv2.INTER_AREA)
    projectScale = 500
    # projectScale = round(panoImage.shape[0] * 2 / np.pi)  # pano -> project 倍数 pi/2
    minLineSquare = 625  # projectScale/10 **2

    projectImageAndMappings = Projection.CubeProjection(panoImage, projectScale)

    panoImageVpsAll = np.full((panoImage.shape[:2]), 2, dtype=np.uint8)  # 存储所有消失点投票
    panoImageVpsGood = np.full((panoImage.shape[:2]), 2, dtype=np.uint8)  # 存储所有消失点投票
    panoImageVpsBad = np.full((panoImage.shape[:2]), 2, dtype=np.uint8)  # 存储所有消失点投票
    panoImageLines = np.zeros(panoImage.shape, dtype=np.uint8)  # 存储直线

    for i in range(len(projectImageAndMappings)):
        print(i)
        projectImage = projectImageAndMappings[i][0]
        mapping = projectImageAndMappings[i][1]
        # TODO step1. 检测直线
        grayImage = cv2.cvtColor(projectImage, cv2.COLOR_BGR2GRAY)
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(grayImage)
        if lines is None:
            continue
        goodLines = []
        for line in lines:
            x0 = line[0][0]
            y0 = line[0][1]
            x1 = line[0][2]
            y1 = line[0][3]
            # TODO 筛选出足够长的线段
            if (x1 - x0) ** 2 + (y1 - y0) ** 2 < minLineSquare:
                continue
            goodLines.append(line)
            # TODO step2. 从投影 xy 坐标映射回全景图 xy 坐标
            x0, y0 = mapping(x0, y0)
            x1, y1 = mapping(x1, y1)
            # TODO step3. 由线段两个端点和球心计算直线所在大圆的 normal
            xa, ya, za = CoordsTransfrom.xy2xyz(x0, y0, panoImage.shape)
            xb, yb, zb = CoordsTransfrom.xy2xyz(x1, y1, panoImage.shape)
            a = np.array([xa, ya, za])
            b = np.array([xb, yb, zb])
            normal = np.cross(a, b)
            # TODO step4. normal 垂直的大圆变换回 xy 并存储进 hough 图像
            # TODO 可以 Hough 投票检测到 4 个消失点
            v1, v2 = Projection.BuildCoords(normal)
            for theta in np.linspace(0, 2 * np.pi, samples, endpoint=False):
                p = np.cos(theta) * v1 + np.sin(theta) * v2
                xp, yp = CoordsTransfrom.xyz2xy(p[0], p[1], p[2], panoImage.shape)
                panoImageVpsAll[yp][xp] *= 4

                # TODO step5. 抑制线上的投票点，p 与 a,b 叉乘，同向的点保留，不同向的点为线上点，或线上点的对称点，不予保留，都与 normal 方向相同或者相反即为同向
                pa = np.cross(p, a)
                pb = np.cross(p, b)

                if np.dot(pa, normal) * np.dot(pb, normal) < 0:
                    panoImageVpsBad[yp][xp] *= 4
                    continue

                pa = np.cross(p, -a)
                pb = np.cross(p, -b)

                if np.dot(pa, normal) * np.dot(pb, normal) < 0:
                    panoImageVpsBad[yp][xp] *= 4
                    continue

                panoImageVpsGood[yp][xp] *= 4

        Visualization.DrawPanoLine(panoImageLines, goodLines, mapping, (0, 255, 0), sampleRate=1.1)

    cv2.imwrite("output_test_panoImageLines_" + str(logid) + ".jpg", panoImageLines)
    cv2.imwrite("output_test_panoImageVpsAll_" + str(logid) + ".jpg", panoImageVpsAll)
    cv2.imwrite("output_test_panoImageVpsGood_" + str(logid) + ".jpg", panoImageVpsGood)
    cv2.imwrite("output_test_panoImageVpsBad_" + str(logid) + ".jpg", panoImageVpsBad)
    panoImageVpsGood_median = cv2.medianBlur(panoImageVpsGood, 5)
    cv2.imwrite("output_test_panoImageVpsGood_median_5_" + str(logid) + ".jpg", panoImageVpsGood_median)


def HoughGreatCircleVpEstimationV2(panoImage, projectImageAndMappings, samples=360):
    panoImageWithGreatCircleNormal = np.full((panoImage.shape[:2]), 2, dtype=np.uint8)  # 存储所有直线所在大圆的法线方向
    panoImageEdges = np.zeros(panoImage.shape, dtype=np.uint8)  # 存储直线
    for i in range(len(projectImageAndMappings)):
        print(i)
        projectImage = projectImageAndMappings[i][0]
        mapping = projectImageAndMappings[i][1]
        # TODO step1. 检测直线
        grayImage = cv2.cvtColor(projectImage, cv2.COLOR_BGR2GRAY)
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(grayImage)
        if lines is None:
            continue
        Visualization.DrawPanoLine(panoImageEdges, lines, mapping, (0, 255, 0), sampleRate=1.1)
        for line in lines:
            x0 = line[0][0]
            y0 = line[0][1]
            x1 = line[0][2]
            y1 = line[0][3]
            # TODO step2. 从投影 xy 坐标映射回全景图 xy 坐标
            x0, y0 = mapping(x0, y0)
            x1, y1 = mapping(x1, y1)
            # TODO step3. 由线段两个端点和球心计算直线所在大圆的 normal
            xa, ya, za = CoordsTransfrom.xy2xyz(x0, y0, panoImage.shape)
            xb, yb, zb = CoordsTransfrom.xy2xyz(x1, y1, panoImage.shape)
            a = np.array([xa, ya, za])
            b = np.array([xb, yb, zb])
            normal = np.cross(a, b)
            # TODO step4. normal 垂直的大圆变换回 xy 并存储进 hough 图像
            # TODO 可以 Hough 投票检测到 4 个消失点
            v1, v2 = Projection.BuildCoords(normal)
            for theta in np.linspace(0, 2 * np.pi, samples, endpoint=False):
                p = np.cos(theta) * v1 + np.sin(theta) * v2

                # TODO step5. 抑制线上的投票点，p 与 a,b 叉乘，同向的点保留，不同向的点为线上点，或线上点的对称点，不予保留，都与 normal 方向相同或者相反即为同向
                pa = np.cross(p, a)
                pb = np.cross(p, b)

                if np.dot(pa, normal) * np.dot(pb, normal) < 0:
                    continue

                pa = np.cross(p, -a)
                pb = np.cross(p, -b)

                if np.dot(pa, normal) * np.dot(pb, normal) < 0:
                    continue

                xp, yp = CoordsTransfrom.xyz2xy(p[0], p[1], p[2], panoImage.shape)
                panoImageWithGreatCircleNormal[yp][xp] *= 4

    # vps = np.where(panoImageWithGreatCircleNormal == max(panoImageWithGreatCircleNormal.flat))
    # for i in range(len(vps[0])):
    #     x = vps[1][i]
    #     y = vps[0][i]
    #     cv2.circle(panoImageWithGreatCircleNormal, (x, y), 10, 255, -1)

    return panoImageEdges, panoImageWithGreatCircleNormal
    # cv2.imwrite("output_hough_great_circle_normal_p.jpg", panoImageWithGreatCircleNormal)


def HoughGreatCircleVpEstimation(panoImage, projectImageAndMappings, samples=360):
    panoImageWithGreatCircleNormal = np.zeros((panoImage.shape[:2]), dtype=np.uint8)  # 存储所有直线所在大圆的法线方向
    for i in range(len(projectImageAndMappings)):
        print(i)
        projectImage = projectImageAndMappings[i][0]
        mapping = projectImageAndMappings[i][1]
        # TODO step1. 检测直线
        grayImage = cv2.cvtColor(projectImage, cv2.COLOR_BGR2GRAY)
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(grayImage)
        if lines is None:
            continue
        for line in lines:
            x0 = line[0][0]
            y0 = line[0][1]
            x1 = line[0][2]
            y1 = line[0][3]
            # TODO step2. 从投影 xy 坐标映射回全景图 xy 坐标
            x0, y0 = mapping(x0, y0)
            x1, y1 = mapping(x1, y1)
            # TODO step3. 由线段两个端点和球心计算直线所在大圆的 normal
            xa, ya, za = CoordsTransfrom.xy2xyz(x0, y0, panoImage.shape)
            xb, yb, zb = CoordsTransfrom.xy2xyz(x1, y1, panoImage.shape)
            a = np.array([xa, ya, za])
            b = np.array([xb, yb, zb])
            normal = np.cross(a, b)
            # # TODO step4. normal 和 -norml 变换回 xy 并存储进 hough 图像
            # TODO 可以由大圆检测到 6 个消失点
            # xn, yn = CoordsTransfrom.xyz2xy(normal[0], normal[1], normal[2], panoImage.shape)
            # x_n, y_n = CoordsTransfrom.xyz2xy(-normal[0], -normal[1], -normal[2], panoImage.shape)
            # cv2.circle(panoImageWithGreatCircleNormal, (xn, yn), 10, 255, -1)
            # cv2.circle(panoImageWithGreatCircleNormal, (x_n, y_n), 10, 255, -1)
            # TODO step4. normal 垂直的大圆变换回 xy 并存储进 hough 图像
            # TODO 可以 Hough 投票检测到 4 个消失点
            v1, v2 = Projection.BuildCoords(normal)
            for theta in np.linspace(0, 2 * np.pi, samples, endpoint=False):
                p = np.cos(theta) * v1 + np.sin(theta) * v2
                xp, yp = CoordsTransfrom.xyz2xy(p[0], p[1], p[2], panoImage.shape)
                panoImageWithGreatCircleNormal[yp][xp] += 40

    # vps = np.where(panoImageWithGreatCircleNormal == max(panoImageWithGreatCircleNormal.flat))
    # for i in range(len(vps[0])):
    #     x = vps[1][i]
    #     y = vps[0][i]
    #     cv2.circle(panoImageWithGreatCircleNormal, (x, y), 10, 255, -1)

    return panoImageWithGreatCircleNormal
    # cv2.imwrite("output_hough_great_circle_normal_p.jpg", panoImageWithGreatCircleNormal)


def HoughVpEstimation(projectImage, scaleT=(-15, 16), sampleRate=1.1):
    """
    Hough 算法检测灭点
    :param projectImage:    检测的透视投影图
    :return:                灭点坐标（x,y）
    """
    # TODO step1. 检测直线
    grayImage = cv2.cvtColor(projectImage, cv2.COLOR_BGR2GRAY)
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(grayImage)
    houghImage = np.zeros(shape=grayImage.shape, dtype=np.uint8)
    # TODO step2. 延长直线
    goodLines = []
    for line in lines:
        x0 = line[0][0]
        y0 = line[0][1]
        x1 = line[0][2]
        y1 = line[0][3]

        dx = x1 - x0
        dy = y1 - y0

        if abs(dx) < 10 or abs(dy) < 10 or dx ** 2 + dy ** 2 < 100:
            # print("small", dy, dx, line)
            continue
        # else: # 删除不影响稳定性
        #     k = dy / dx
        #     if abs(k) > 30 or abs(k) < 0.03:
        #         continue
        goodLines.append(line)

        samples = round(max(abs(dx), abs(dy)) * sampleRate * (scaleT[1] - scaleT[0]))

        for dt in np.linspace(scaleT[0], scaleT[1], samples):
            x = x0 + dt * dx
            y = y0 + dt * dy
            ix = round(x)
            iy = round(y)
            if ix > 0 and ix < houghImage.shape[1] and iy > 0 and iy < houghImage.shape[1]:
                houghImage[iy][ix] += 30
                projectImage[iy][ix] = [0, 255, 0]
    houghImageUniqueFlat = np.sort(np.unique(houghImage.flat))
    # print(houghImageUniqueFlat)
    # vps = np.where(
    #     (houghImage == houghImageUniqueFlat[len(houghImageUniqueFlat) - 1]) | (houghImage == houghImageUniqueFlat[
    #         len(houghImageUniqueFlat) - 2]))
    vps = np.where(houghImage == houghImageUniqueFlat[len(houghImageUniqueFlat) - 1])

    # print(np.max(houghImage), vps)
    counts = []
    for i in range(len(vps[0])):
        x = vps[1][i]
        y = vps[0][i]
        count = 0
        for line in goodLines:
            x0 = line[0][0]
            y0 = line[0][1]
            x1 = line[0][2]
            y1 = line[0][3]

            dx = x1 - x0
            dy = y1 - y0

            k = dy / dx
            b = y0 - k * x0
            err = y - k * x + b
            if err < 0.1:
                count += 1
        print(count)
        counts.append(count)
    for i in range(len(vps[0])):
        cv2.circle(houghImage, (vps[1][i], vps[0][i]), 10, 100, -1)
        cv2.circle(projectImage, (vps[1][i], vps[0][i]), 10, (0, 255, 0), -1)
    Confidence = max(counts) / len(goodLines)
    print(Confidence, np.array(counts) / len(goodLines))  # 置信度要高于0.3
    vpIndex = counts.index(max(counts))
    cv2.circle(houghImage, (vps[1][vpIndex], vps[0][vpIndex]), 10, 255, -1)
    cv2.circle(projectImage, (vps[1][vpIndex], vps[0][vpIndex]), 30, (0, 255, 0), -1)
    # cv2.imshow("vp", projectImage)
    # cv2.imwrite("output_vp_" + str(time.time()) + ".jpg", projectImage)
    # cv2.waitKey(0)
    return True, vps[1][vpIndex], vps[0][vpIndex]


def __ransacVpEstimationSelectSamples(sampleAll):
    """
    选择 2 个不同的随机数作为样本
    """
    samples = []
    for i in range(2):
        while True:
            sample = int(random.random() * sampleAll)
            if sample not in samples:
                samples.append(sample)
                break
    print(samples)
    time.sleep(0.1)
    return samples


def __ransacVpEstimationComputeParams(lines, samples):
    """
    根据选择的样本计算灭点
    """
    x00 = lines[samples[0]][0][0]
    y00 = lines[samples[0]][0][1]
    x01 = lines[samples[0]][0][2]
    y01 = lines[samples[0]][0][3]

    x10 = lines[samples[1]][0][0]
    y10 = lines[samples[1]][0][1]
    x11 = lines[samples[1]][0][2]
    y11 = lines[samples[1]][0][3]

    k0 = (y01 - y00) / (x01 - x00)
    b0 = y00 - k0 * x00
    k1 = (y11 - y10) / (x11 - x10)
    b1 = y10 - k1 * x10

    x = (b0 - b1) / (k1 - k0)
    y = x * k0 + b0

    return x, y


def __ransacVpEstimationSelectSamplesAndComputeParams(lines):
    """
    选择 2 个不同的随机数作为样本，根据选择的样本计算灭点
    """
    epsilon = 0.01
    x, y = 0, 0
    while True:
        # 随机选择两个样本
        samples = []
        for i in range(2):
            sample = int(random.random() * len(lines))
            samples.append(sample)

        x00 = lines[samples[0]][0][0]
        y00 = lines[samples[0]][0][1]
        x01 = lines[samples[0]][0][2]
        y01 = lines[samples[0]][0][3]

        x10 = lines[samples[1]][0][0]
        y10 = lines[samples[1]][0][1]
        x11 = lines[samples[1]][0][2]
        y11 = lines[samples[1]][0][3]

        # 与 y 轴平行，重新选择
        if abs(x01 - x00) < epsilon or abs(x11 - x10) < epsilon:
            continue
        k0 = (y01 - y00) / (x01 - x00)
        b0 = y00 - k0 * x00
        k1 = (y11 - y10) / (x11 - x10)
        b1 = y10 - k1 * x10

        # 两直线平行，重新选择
        if abs(k0 - k1) < epsilon:
            continue

        # 计算出灭点，终止循环
        x = (b0 - b1) / (k1 - k0)
        y = x * k0 + b0
        break

    return x, y


def __ransacVpEstimationGetInlierSamples(lines, x, y, inlierErr=20):
    """
    根据计算的灭点数据筛选内点
    """
    epsilon = 0.01
    inlierSamples = []
    for line in lines:
        # print(line)
        # inlierSamples.append(line)
        # print(inlierSamples)
        # exit(-1)
        # continue
        x0 = line[0][0]
        y0 = line[0][1]
        x1 = line[0][2]
        y1 = line[0][3]

        if abs(x1 - x0) < epsilon:
            err = x - x0
        else:
            k = (y1 - y0) / (x1 - x0)
            b = y0 - k * x0
            err = y - k * x + b

        if abs(err) < inlierErr:
            inlierSamples.append(line)
            print(line, err, len(inlierSamples))

    # print("inlierSamples", len(inlierSamples))
    return inlierSamples


def RansacVpEstimation(projectImage, iterations=10, ransacRate=0.1):
    """
    Ransac 算法检测灭点
    :param projectImage:    检测的透视投影图
    :param iterations:      迭代次数
    :param ransacRate:      内点的比例
    :return:                灭点坐标（x,y）
    """
    # TODO step1. 检测直线
    gray = cv2.cvtColor(projectImage, cv2.COLOR_BGR2GRAY)
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(gray)
    # projectImage = fld.drawSegments(projectImage, lines)
    # TODO step2. Ransac 检测灭点
    sampleAll = len(lines)
    it = 0
    while True:
        print(len(lines), sampleAll)
        while True:
            # TODO step2.1. 随机选择样本
            # TODO step2.2. 根据选择的样本计算灭点
            x, y = __ransacVpEstimationSelectSamplesAndComputeParams(lines)
            # TODO step2.3. 直到计算的灭点在透视投影图内（这样才能映射到全景图上）
            if x > 0 and x < projectImage.shape[1] and y > 0 and y < projectImage.shape[0]:
                break
            # print(len(lines), x, y)
        # TODO step2.4. 由计算的灭点数据确定内点
        inlierSamples = __ransacVpEstimationGetInlierSamples(lines, x, y)
        # TODO step2.5. 内点大于一定比率时更新内点为待检测数据，开始下一次迭代
        if len(inlierSamples) > sampleAll * ransacRate:
            print(len(inlierSamples), sampleAll * ransacRate)
            print(x, y)
            # print(inlierSamples.shape)
            # print(lines.shape)
            # print(lines)
            # print(inlierSamples)
            inlierSamplesImage = projectImage.copy()
            inlierSamplesImage = fld.drawSegments(inlierSamplesImage, lines)
            for line in inlierSamples:
                x0 = line[0][0]
                y0 = line[0][1]
                x1 = line[0][2]
                y1 = line[0][3]
                cv2.line(inlierSamplesImage, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(inlierSamplesImage, (round(x), round(y)), 20, (0, 0, 0), -1)
            cv2.imshow("vp", inlierSamplesImage)
            cv2.waitKey(0)
            lines = inlierSamples
            it += 1
            sampleAll *= 0.8
        # TODO step2.6. 达到一定迭代次数后终止循环
        if it == iterations:
            break
    return x, y


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
