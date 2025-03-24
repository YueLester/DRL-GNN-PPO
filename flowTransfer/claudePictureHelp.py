import numpy as np
import matplotlib.pyplot as plt

# 创建示例数据
x = np.linspace(0, 10, 100)
y1_curves = [np.sin(x + i) for i in range(5)]
y2_curves = [np.cos(x + i) * np.exp(-x/5) for i in range(5)]
y3_curves = [np.exp(-x/3 + i/2) for i in range(5)]
y4_curves = [(x + i)**2 / 50 for i in range(5)]

# 创建图形和子图
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
fig.subplots_adjust(hspace=0.1)  # 调整子图间距

# 设置不同的颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 第一个子图
for i, y in enumerate(y1_curves):
    ax1.plot(x, y, label=f'Curve {i+1}', color=colors[i])
ax1.set_ylabel('Values 1')
ax1.set_ylim(-2, 2)  # 自定义y轴范围
ax1.legend()
ax1.grid(True)

# 第二个子图
for i, y in enumerate(y2_curves):
    ax2.plot(x, y, label=f'Curve {i+1}', color=colors[i])
ax2.set_ylabel('Values 2')
ax2.set_ylim(-1, 1)  # 自定义y轴范围
ax2.legend()
ax2.grid(True)

# 第三个子图
for i, y in enumerate(y3_curves):
    ax3.plot(x, y, label=f'Curve {i+1}', color=colors[i])
ax3.set_ylabel('Values 3')
ax3.set_ylim(0, 1.5)  # 自定义y轴范围
ax3.legend()
ax3.grid(True)

# 第四个子图
for i, y in enumerate(y4_curves):
    ax4.plot(x, y, label=f'Curve {i+1}', color=colors[i])
ax4.set_xlabel('X axis')
ax4.set_ylabel('Values 4')
ax4.set_ylim(0, 5)  # 自定义y轴范围
ax4.legend()
ax4.grid(True)

# 添加总标题
plt.suptitle('Multiple Curves with Shared X-axis', fontsize=16)

# 显示图形
plt.show()