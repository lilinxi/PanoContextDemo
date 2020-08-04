import imageio as im
import numpy as np
import open3d
import CoordsTransfrom
import cv2

# img = im.imread('./output_pano_lsd.jpg')
img = cv2.imread('../images/vp.png')
img = np.array(img)
print(img.shape)

points = []
colors = []
for i in range(0, img.shape[0], 1):
    for j in range(0, img.shape[1], 1):
        u = i / img.shape[0]
        v = j / img.shape[1]
        rgb = img[i][j] / 255
        x, y, z = CoordsTransfrom.uv2xyz(u, v)
        points.append([x, y, z])
        colors.append(rgb)

points = np.array(points)
colors = np.array(colors)
print(points.shape)
print(colors.shape)

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(points)
pcd.colors = open3d.utility.Vector3dVector(colors)
open3d.visualization.draw_geometries([pcd])
