import numpy as np


def SolidAngleSegmentation(segmentationShape=(10, 20)):
    """
    dw = sin(phi) * d(theta) * d(phi)
    f(phi_1 -> phi_0) = -(cos(phi_1) - cos(phi_0)) * d(theta)
    按照 cos(phi) * theta 将 Sphere 划分为等立体角（等面积）的网格 Segmentation
    :param segmentationShape:           划分网格的细粒度
    :return:
    """
    # TODO 每个间隔为一个分段
    cosPhiSegmentation = np.linspace(1, -1, segmentationShape[0] + 1)
    phiSegmentation = np.arccos(cosPhiSegmentation)
    thetaSegmentation = np.linspace(0, 2 * np.pi, segmentationShape[1] + 1)

    # for i in range(segmentationShape[0]):
    #     for j in range(segmentationShape[1]):
    #         w = -(np.cos(phiSegmentation[i + 1]) - np.cos(phiSegmentation[i])) * \
    #             (thetaSegmentation[j + 1] - thetaSegmentation[j])
    #         print(i, j, w)  # w 应相同

    def phi_theta2segmentation(phi, theta):
        phi_index = 0
        theta_index = 0
        for i in range(1, segmentationShape[0]):
            if phiSegmentation[i] > phi:
                break
            else:
                phi_index += 1

        for i in range(1, segmentationShape[1]):
            if thetaSegmentation[i] > theta:
                break
            else:
                theta_index += 1

        return phi_index, theta_index

    pass


if __name__ == "__main__":
    SolidAngleSegmentation()
