import numpy as np
import matplotlib.pyplot as plt

def generate_spiral_data(n_samples=1000, noise=0.2, n_rotations=2):
    """
    生成双螺旋分类数据
    
    参数:
    n_samples: 每个类别的样本数
    noise: 噪声水平
    n_rotations: 螺旋的圈数
    
    返回:
    X: 形状为 (n_samples*2, 2) 的数组，包含两个特征
    y: 形状为 (n_samples*2,) 的数组，包含类别标签 (0 或 1)
    """
    np.random.seed(42)
    
    # 生成螺旋参数
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi * n_rotations
    
    # 第一个螺旋（类别0）
    r_a = theta + np.pi
    x_a = r_a * np.cos(theta)
    y_a = r_a * np.sin(theta)
    
    # 第二个螺旋（类别1）
    r_b = theta + np.pi
    x_b = r_b * np.cos(theta + np.pi)
    y_b = r_b * np.sin(theta + np.pi)
    
    # 添加噪声
    x_a += np.random.randn(n_samples) * noise
    y_a += np.random.randn(n_samples) * noise
    x_b += np.random.randn(n_samples) * noise
    y_b += np.random.randn(n_samples) * noise
    
    # 组合数据
    X = np.vstack([np.column_stack((x_a, y_a)), 
                   np.column_stack((x_b, y_b))])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    return X, y

# 生成数据
X, y = generate_spiral_data(n_samples=1000, noise=0.2, n_rotations=2)

# 创建图形
plt.figure(figsize=(10, 10))

# 绘制两个类别的点
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', alpha=0.5)

plt.title('Two-Class Spiral Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.axis('equal')

# 设置坐标轴范围
max_range = max(abs(X.min()), abs(X.max())) * 1.1
plt.xlim(-max_range, max_range)
plt.ylim(-max_range, max_range)

# 保存图片
plt.savefig('spiral_classification_data.png')
plt.close()

print(f"数据集大小: {X.shape}")
print(f"类别 0 的样本数: {np.sum(y == 0)}")
print(f"类别 1 的样本数: {np.sum(y == 1)}")
