import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import subprocess
import shutil

# 创建保存帧的文件夹
os.makedirs('rotation_frames', exist_ok=True)

def generate_spiral_data_3d(n_samples=1000, noise=0.2, n_rotations=3):
    """生成3D双螺旋分类数据"""
    np.random.seed(42)
    
    # 生成螺旋参数
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi * n_rotations
    
    # 第一个螺旋（类别0）
    r_a = theta + np.pi
    x_a = r_a * np.cos(theta)
    y_a = r_a * np.sin(theta)
    z_a = np.ones(n_samples)  # z=1平面
    
    # 第二个螺旋（类别1）
    r_b = theta + np.pi
    x_b = r_b * np.cos(theta + np.pi)
    y_b = r_b * np.sin(theta + np.pi)
    z_b = -np.ones(n_samples)  # z=-1平面
    
    # 添加噪声
    x_a += np.random.randn(n_samples) * noise
    y_a += np.random.randn(n_samples) * noise
    x_b += np.random.randn(n_samples) * noise
    y_b += np.random.randn(n_samples) * noise
    
    # 组合数据
    X = np.vstack([np.column_stack((x_a, y_a, z_a)), 
                   np.column_stack((x_b, y_b, z_b))])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    return X, y

# 生成3D数据
X, y = generate_spiral_data_3d(n_samples=1000, noise=0.2, n_rotations=3)

# 设置固定的图形大小
plt.rcParams['figure.figsize'] = [12, 12]

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制两个类别的点
scatter0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], 
                     c='blue', label='Class 0', alpha=0.6)
scatter1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], 
                     c='red', label='Class 1', alpha=0.6)

# 添加标题和标签
ax.set_title('3D Two-Class Spiral Dataset', fontsize=16, pad=20)
ax.set_xlabel('X', fontsize=12, labelpad=10)
ax.set_ylabel('Y', fontsize=12, labelpad=10)
ax.set_zlabel('Z', fontsize=12, labelpad=10)

# 添加图例
ax.legend(fontsize=12)

# 设置网格
ax.grid(True)

# 设置视角
ax.view_init(elev=20, azim=45)

# 保存静态图片
plt.savefig('spiral_classification_3d.png', dpi=300, bbox_inches='tight')

# 创建动画效果（旋转视角）
angles = np.linspace(0, 360, 120)

print("正在生成动画...")
# 创建动画帧
for i, angle in enumerate(angles):
    ax.view_init(elev=20, azim=angle)
    plt.savefig(f'rotation_frames/frame_{i:03d}.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1)
    if i % 10 == 0:
        print(f"已完成 {i/len(angles)*100:.1f}%")

# 使用subprocess执行ffmpeg命令
try:
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', '30',
        '-i', 'rotation_frames/frame_%03d.png',
        '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        'spiral_3d_rotation.mp4'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("FFmpeg错误:")
        print(result.stderr)
    else:
        print("视频生成成功！")
except FileNotFoundError:
    print("错误：找不到ffmpeg。请确保已安装ffmpeg并添加到系统路径。")

plt.close()

print("完成！生成了：")
print("1. spiral_classification_3d.png - 静态3D图")
if os.path.exists('spiral_3d_rotation.mp4') and os.path.getsize('spiral_3d_rotation.mp4') > 0:
    print("2. spiral_3d_rotation.mp4 - 3D旋转动画")
else:
    print("警告：视频生成失败")

# # 清理临时文件
# shutil.rmtree('rotation_frames')
