import cv2

img = cv2.imread('output_test_panoImageVpsGood_0.jpg', cv2.IMREAD_GRAYSCALE)
print(img.shape)
img_median = cv2.medianBlur(img, 7)
cv2.imwrite("output_test_panoImageVpsGood_0_median_7.jpg", img_median)
