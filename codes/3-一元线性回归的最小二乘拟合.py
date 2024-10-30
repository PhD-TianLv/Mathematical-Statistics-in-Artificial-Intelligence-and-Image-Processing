import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 生成基础数据点
X = np.concatenate([
    np.random.normal(1, 0.5, 100),    # 在x=1附近集中一些点
    np.random.normal(4, 0.8, 300),    # 在x=4附近集中更多点
    np.random.uniform(0, 5, 100)      # 均匀分布的点
])

# 真实参数
true_w = 2.5
true_b = 4.0

# 生成y值，加入不同程度的噪声
noise = np.random.normal(0, 0.5, len(X))

y = true_w * X + true_b + noise

# 最小二乘法计算
X_mean = np.mean(X)
y_mean = np.mean(y)

# 计算w（斜率）
numerator = np.sum((X - X_mean) * (y - y_mean))
denominator = np.sum((X - X_mean) ** 2)
w = numerator / denominator

# 计算b（截距）
b = y_mean - w * X_mean

# 预测值
y_pred = w * X + b

# 计算R方（拟合优度）
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_tot)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='blue', alpha=0.5, label='Sample Points')
plt.plot(X, true_w * X + true_b, 'r-', label='Ground Truth')
plt.plot(X, y_pred, 'g--', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Least Squares Fitting (w={w:.2f}, b={b:.2f}, R²={r2:.3f})')
plt.legend()
plt.grid(True)
plt.show()

print(f'Estimated parameters: w = {w:.4f}, b = {b:.4f}')
print(f'True parameters: w = {true_w}, b = {true_b}')
print(f'R-squared: {r2:.4f}')
