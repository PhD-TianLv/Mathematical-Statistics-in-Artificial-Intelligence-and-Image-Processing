import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('猫_分割后.png', cv2.IMREAD_GRAYSCALE)

# 计算原始矩
moments = cv2.moments(image)

# 计算质心
if moments['m00'] != 0:
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])
else:
    centroid_x, centroid_y = 0, 0

# 可视化图像并绘制质心
fig, ax = plt.subplots()
ax.imshow(image, cmap="gray")
ax.axis('off')

# 在图像上绘制质心点
plt.plot(centroid_x, centroid_y, 'ro')  # 红色的点

# 显示质心坐标
plt.text(centroid_x + 10, centroid_y, f'({centroid_x}, {centroid_y})', color='red', fontsize=12)

plt.savefig('猫_质心.png', bbox_inches='tight', pad_inches=0, transparent=True)
plt.close()

