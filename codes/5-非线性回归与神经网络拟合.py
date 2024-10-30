import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm  # 添加这一行
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpiralParametricNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, t):
        return self.net(t)

def generate_spiral_data(n_samples=1000):
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, n_samples)
    r = np.linspace(0.5, 2, n_samples) + np.random.normal(0, 0.1, n_samples)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return t, x, y

# 准备数据
t, x, y = generate_spiral_data()
t_tensor = torch.FloatTensor(t.reshape(-1, 1)).to(device)
xy_tensor = torch.FloatTensor(np.stack([x, y], axis=1)).to(device)

# 创建模型和优化器
model = SpiralParametricNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练参数
epochs = 200

# 创建动画
fig, ax = plt.subplots(figsize=(10, 10))

def init():
    ax.clear()
    ax.scatter(x, y, alpha=0.5, label='Data Points')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Spiral Fitting (epoch 0)')
    ax.legend()
    ax.grid(True)
    return []

def update(frame):
    # 使用tqdm显示进度
    if frame == 1:  # 只在第一帧初始化进度条
        update.pbar = tqdm(total=epochs, desc='Training')
    elif frame > 1:  # 从第二帧开始更新进度条
        update.pbar.update(1)
    elif frame == epochs:  # 在最后一帧关闭进度条
        update.pbar.close()
    
    # 训练一个epoch
    epoch_loss = 0
    batch_count = 0
    
    if frame > 0:
        dataset = torch.utils.data.TensorDataset(t_tensor, xy_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        
        for batch_t, batch_xy in dataloader:
            optimizer.zero_grad()
            pred_xy = model(batch_t)
            loss = criterion(pred_xy, batch_xy)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
    
    # 绘图
    ax.clear()
    ax.scatter(x, y, alpha=0.5, label='Data Points')
    
    # 绘制预测曲线
    with torch.no_grad():
        t_test = torch.linspace(0, 4*np.pi, 1000).reshape(-1, 1).to(device)
        xy_pred = model(t_test)
        x_pred, y_pred = xy_pred[:, 0].cpu().numpy(), xy_pred[:, 1].cpu().numpy()
        ax.plot(x_pred, y_pred, 'r-', label='Fitted Curve')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if frame > 0:
        avg_loss = epoch_loss / batch_count
        ax.set_title(f'Spiral Fitting (epoch {frame}, loss: {avg_loss:.6f})')
    else:
        ax.set_title('Spiral Fitting (epoch 0)')
    ax.legend()
    ax.grid(True)
    
    return []

print("Starting training...")
# 创建动画
anim = FuncAnimation(fig, update, frames=epochs+1,
                    init_func=init, blit=True,
                    interval=50, repeat=False)

print("Creating animation...")
# 保存动画
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
anim.save('spiral_fitting_nn.mp4', writer=writer)
plt.close()

print("Animation saved as 'spiral_fitting_nn.mp4'")
