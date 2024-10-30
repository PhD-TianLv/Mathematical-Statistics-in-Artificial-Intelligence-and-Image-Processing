import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# 使用与之前相同的数据生成方式
np.random.seed(42)
X = np.concatenate([
    np.random.normal(1, 0.5, 100),
    np.random.normal(4, 0.8, 300),
    np.random.uniform(0, 5, 100)
])

true_w = 2.5
true_b = 4.0
noise = np.random.normal(0, 0.5, len(X))
y = true_w * X + true_b + noise

# 梯度下降参数
learning_rate = 0.05
epochs = 200
w = 0.0  # 初始权重
b = 12.5  # 初始偏置

# 存储每一步的参数
w_history = []
b_history = []

# 创建动画
fig, ax = plt.subplots(figsize=(10, 6))

def init():
    ax.clear()
    ax.scatter(X, y, c='blue', alpha=0.5, label='Sample Points')
    ax.plot(X, true_w * X + true_b, 'r-', label='Ground Truth')
    ax.plot(X, w * X + b, 'g--', label='Current Fit')  # 添加初始拟合线
    
    ax.set_xlim(min(X)-0.5, max(X)+0.5)
    ax.set_ylim(min(y)-1, max(y)+1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Gradient Descent (0/{epochs} w={w:.2f}, b={b:.2f})')  # 添加初始标题
    ax.legend()
    ax.grid(True)
    return []

def update(frame):
    global w, b
    
    if frame > 0:  # 只有在frame>0时才更新参数
        # 计算梯度
        y_pred = w * X + b
        dw = -2 * np.mean(X * (y - y_pred))
        db = -2 * np.mean(y - y_pred)
        
        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        w_history.append(w)
        b_history.append(b)
    
    # 绘制
    ax.clear()
    ax.scatter(X, y, c='blue', alpha=0.5, label='Sample Points')
    ax.plot(X, true_w * X + true_b, 'r-', label='Ground Truth')
    ax.plot(X, w * X + b, 'g--', label='Current Fit')
    
    ax.set_xlim(min(X)-0.5, max(X)+0.5)
    ax.set_ylim(min(y)-1, max(y)+1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Gradient Descent ({frame}/{epochs} w={w:.2f}, b={b:.2f})')  # 修改frame+1为frame
    ax.legend()
    ax.grid(True)
    
    return []

# 创建动画时增加frames数量
anim = FuncAnimation(fig, update, frames=epochs+1,  # 修改为epochs+1
                    init_func=init, blit=True, 
                    interval=5, repeat=False)

# 修改保存动画的部分
writer = FFMpegWriter(
    fps=30,  # 帧率
    metadata=dict(artist='Me'),
    bitrate=1800  # 比特率
)

# 保存为MP4
anim.save('gradient_descent.mp4', writer=writer)
# plt.show()

print(f'Final parameters: w = {w:.4f}, b = {b:.4f}')
print(f'True parameters: w = {true_w}, b = {true_b}')
