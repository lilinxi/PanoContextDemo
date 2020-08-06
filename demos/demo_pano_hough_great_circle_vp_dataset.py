import cv2
import Projection
import VpEstimation

for i in range(50, 88):
    panoImage = cv2.imread("../dataset/dataset-" + str(i) + ".jpg", cv2.IMREAD_COLOR)
    projectScale = round(panoImage.shape[0] / 3)

    print(panoImage.shape)

    projectImageAndMappings = Projection.CubeProjection(panoImage, projectScale)

    panoImageEdges, panoImageWithGreatCircleNormal = \
        VpEstimation.HoughGreatCircleVpEstimationV2(panoImage,projectImageAndMappings)

    cv2.imwrite("output_dataset-vp-weight-v2-" + str(i) + ".jpg", panoImageWithGreatCircleNormal)
    cv2.imwrite("output_dataset-edges-" + str(i) + ".jpg", panoImageEdges)
