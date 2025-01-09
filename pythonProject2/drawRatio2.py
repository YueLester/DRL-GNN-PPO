import random

from matplotlib import pyplot as plt
# 示例数据
from matplotlib.ticker import MaxNLocator


nodeNum = [11, 15, 19, 23, 27, 31]

x = [30, 30, 30, 30, 30, 30]  # x轴的数据点
y1 = [9, 10, 13, 15, 18, 18]
y2 = [6, 8, 11, 13, 13, 16]
y3 = [3, 5, 7, 8, 10, 11]

# r = random.randint(1, 10)

k1 = []
k2 = []
k3 = []
# k1 = [0.8222, 0.918, 0.9344, 0.9262, 0.9371, 0.9353]  # 线1的纵坐标
# k2 = [0.8988, 0.9334, 0.9435, 0.9407, 0.9453, 0.9453]  # 线2的纵坐标
# k3 = [0.8988, 0.9334, 0.9435, 0.9407, 0.9453, 0.9453]  # 线3的纵坐标
for xx, yy in zip(x, y1):
    temp = 1.0 * yy/xx
    k1.append(temp)
for xx, yy in zip(x, y2):
    temp = 1.0 * yy/xx
    k2.append(temp)
for xx, yy in zip(x, y3):
    temp = 1.0 * yy/xx
    k3.append(temp)


x = nodeNum
plt.plot(x, k1, 's-', color='r', label="proposal")  # s-:方形
plt.plot(x, k2, 'o-', color='g', label="greedy")  # o-:圆形
plt.plot(x, k3, '^-', color='b', label="random")  # o-:圆形

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


# ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保主刻度是整数

plt.xlabel("nodeNum")  # ratio
plt.ylabel("ratio")  # 纵坐标名字
plt.legend(loc="best")  # 图例
plt.show()