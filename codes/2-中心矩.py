import cv2
import numpy as np

def calculate_axes_direction(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 转换为二值图
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 计算图像矩
    moments = cv2.moments(binary)

    # 计算质心
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])

    # 计算中心矩
    mu00 = moments['m00']
    mu11 = moments['m11'] - cx * moments['m01']
    mu20 = moments['m20'] - cx * moments['m10']
    mu02 = moments['m02'] - cy * moments['m01']

    # 计算方向角
    theta = 0.5 * np.arctan2(2*mu11/mu00, (mu20 - mu02)/mu00)

    # 创建透明背景的RGBA图像
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    # 将黑色背景设为透明
    result[np.all(result[:, :, :3] == [0, 0, 0], axis=2), 3] = 0

    # 设置轴长度
    rho = min(image.shape[0], image.shape[1]) // 2

    # 计算主轴端点
    dx_major = rho * np.cos(theta)
    dy_major = rho * np.sin(theta)

    # 计算次轴端点 (垂直于主轴)
    dx_minor = 0.3 * rho * np.cos(theta - np.pi/2)
    dy_minor = 0.3 * rho * np.sin(theta - np.pi/2)

    # 绘制次轴
    short_axis = [
        (int(cx-dx_minor), int(cy-dy_minor)),
        (int(cx), int(cy)),
        (int(cx+dx_minor), int(cy+dy_minor))
    ]
    for i in range(len(short_axis)-1):
        cv2.line(result, short_axis[i], short_axis[i+1], (255,0,0,255), 2)
    for pt in short_axis:
        cv2.circle(result, pt, 5, (255,0,0,255), 3)

    # 绘制主轴
    long_axis = [
        (int(cx-dx_major), int(cy-dy_major)),
        (int(cx), int(cy)),
        (int(cx+dx_major), int(cy+dy_major))
    ]
    for i in range(len(long_axis)-1):
        cv2.line(result, long_axis[i], long_axis[i+1], (0,0,255,255), 2)
    for pt in long_axis:
        cv2.circle(result, pt, 5, (0,0,255,255), 3)

    # 绘制质心
    cv2.circle(result, (int(cx),int(cy)), 5, (0,255,0,255), 3)

    return result, cx, cy, theta

def main():
    # 读取分割好的RGB图像
    image = cv2.imread('狗_分割后.png')

    # 计算并生成结果
    result, cx, cy, theta = calculate_axes_direction(image)

    # 保存结果为PNG（带透明通道）
    cv2.imwrite('result.png', result)

    print(f"Centroid: ({cx}, {cy})")
    print(f"Direction angle: {theta*180/np.pi:.2f} degrees")

if __name__ == "__main__":
    main()

