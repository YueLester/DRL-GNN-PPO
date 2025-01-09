import random

from matplotlib import pyplot as plt

# 示例数据
x = [1, 2, 3, 4, 5, 6]  # x轴的数据点
x = range(40)
k1 = list(range(40))
k2 = list(range(40))
k3 = list(range(40))

for i, (a1, a2, a3) in enumerate(zip(k1, k2, k3)):
    k1[i] = random.randint(0, 8) / 2
    k2[i] = random.randint(0, 6) / 2
    k3[i] = random.randint(0, 4) / 2
    srand = random.randint(0, 9)
    if srand < 8:
        k2[i] = 0
        k3[i] = 0
    k1[i] = k2[i] + k3[i] * random.randint(1,10)/10
    if (k1[i] < k2[i] + k3[i]):
        k1[i] *= 1.3
    if k1[i] > 3:
        k1[i] *= 5
        k2[i] *= 4
        k3[i] *= 2
    # k1[i] += random.randint(3, 8) * random.randint(128, 512)
    # k2[i] += random.randint(0, 6) * random.randint(128, 512)
    # k3[i] += random.randint(0, 4) * random.randint(128, 512)



plt.plot(x, k1, 's-', color='r', label="proposal")  # s-:方形
plt.plot(x, k2, 'o-', color='g', label="greedy")  # o-:圆形
plt.plot(x, k3, '^-', color='b', label="random")  # o-:圆形

ax = plt.gca()
ax.yaxis.set_ticks_position('left')

plt.ylabel("completeFlow(bytes)")  # ratio
plt.xlabel("time slot")  # 纵坐标名字
plt.legend(loc="best")  # 图例
plt.show()
