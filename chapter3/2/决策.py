import numpy as np
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# 配置参数
simulation_time = 100  # 模拟总时长
bandwidth_mean_link1 = 100  # 链路1平均带宽 (Mbps)
bandwidth_mean_link2 = 80  # 链路2平均带宽 (Mbps)
packet_loss_initial_link1 = 0.05  # 链路1初始丢包率
packet_loss_initial_link2 = 0.08  # 链路2初始丢包率
total_packets = 50  # 总数据包数量（包括原始和冗余）
probe_threshold = 25  # 探测阈值（时间间隔），增大以减少探测频率

# 初始化数据结构
time_points = np.arange(simulation_time)
bandwidth_link1 = np.zeros(simulation_time)
bandwidth_link2 = np.zeros(simulation_time)
packet_loss_link1 = np.zeros(simulation_time)
packet_loss_link2 = np.zeros(simulation_time)
w1_allocation = np.zeros(simulation_time)
w2_allocation = np.zeros(simulation_time)
redundancy = np.zeros(simulation_time)
original_packets = np.zeros(simulation_time)
probe_link1 = np.zeros(simulation_time, dtype=bool)
probe_link2 = np.zeros(simulation_time, dtype=bool)
last_probe_link1 = -probe_threshold
last_probe_link2 = -probe_threshold

# 生成带宽数据 - 平缓的随机游走
bandwidth_link1[0] = bandwidth_mean_link1
bandwidth_link2[0] = bandwidth_mean_link2
for i in range(1, simulation_time):
    # 更平缓的随机变化
    random_change1 = np.random.normal(0, 3)  # 减小标准差使变化更平缓
    random_change2 = np.random.normal(0, 2)

    # 更平缓的周期性变化
    periodic_change1 = 5 * np.sin(i / 30)  # 减小振幅，增大周期
    periodic_change2 = 4 * np.sin(i / 25 + 1)

    # 更新带宽，设置下限为更高的值，避免接近0
    bandwidth_link1[i] = max(bandwidth_mean_link1 * 0.7,
                             min(bandwidth_mean_link1 * 1.3,
                                 bandwidth_link1[i - 1] + random_change1 + periodic_change1))

    bandwidth_link2[i] = max(bandwidth_mean_link2 * 0.7,
                             min(bandwidth_mean_link2 * 1.3,
                                 bandwidth_link2[i - 1] + random_change2 + periodic_change2))

# 生成丢包率数据 - 使用马尔可夫链模型，增大丢包率范围
packet_loss_link1[0] = packet_loss_initial_link1
packet_loss_link2[0] = packet_loss_initial_link2
for i in range(1, simulation_time):
    # 随机因素
    random_factor1 = np.random.normal(0, 0.02)  # 增大随机性
    random_factor2 = np.random.normal(0, 0.025)

    # 突变概率
    if np.random.random() < 0.03:  # 减少突变概率
        sudden_change1 = np.random.uniform(-0.15, 0.15)  # 增大突变幅度
    else:
        sudden_change1 = 0

    if np.random.random() < 0.04:
        sudden_change2 = np.random.uniform(-0.2, 0.2)
    else:
        sudden_change2 = 0

    # 更新丢包率，允许更高的丢包率
    packet_loss_link1[i] = np.clip(packet_loss_link1[i - 1] + random_factor1 + sudden_change1, 0.01, 0.3)
    packet_loss_link2[i] = np.clip(packet_loss_link2[i - 1] + random_factor2 + sudden_change2, 0.02, 0.4)

# 模拟探测和决策过程
for i in range(simulation_time):
    # 探测决策 - 减少探测频率
    if i - last_probe_link1 >= probe_threshold or packet_loss_link1[i] > 0.25:  # 增大阈值
        probe_link1[i] = True
        last_probe_link1 = i

    if i - last_probe_link2 >= probe_threshold or packet_loss_link2[i] > 0.3:
        probe_link2[i] = True
        last_probe_link2 = i

    # 根据带宽比例分配资源
    total_bandwidth = bandwidth_link1[i] + bandwidth_link2[i]
    w1_ratio = bandwidth_link1[i] / total_bandwidth

    # 根据丢包率决定冗余包数量
    avg_loss_rate = (packet_loss_link1[i] + packet_loss_link2[i]) / 2
    # 根据丢包率设置更合理的冗余策略
    if avg_loss_rate < 0.1:
        redundancy[i] = int(np.ceil(total_packets * 0.1))  # 低丢包率时的基本冗余
    elif avg_loss_rate < 0.2:
        redundancy[i] = int(np.ceil(total_packets * 0.2))  # 中等丢包率
    else:
        redundancy[i] = int(np.ceil(total_packets * 0.3))  # 高丢包率时增加冗余

    # 确保冗余包数量合理
    redundancy[i] = min(redundancy[i], total_packets // 2)

    # 计算原始数据包数量
    original_packets[i] = total_packets - redundancy[i]

    # 分配包到两条链路
    w1_allocation[i] = int(np.round((original_packets[i] + redundancy[i]) * w1_ratio))
    w2_allocation[i] = total_packets - w1_allocation[i]

# 创建三张扁平的静态图表
plt.figure(figsize=(18, 10))

# 图1：链路带宽和分配资源 - 使用双Y轴
plt.subplot(3, 1, 1)
fig1 = plt.gca()
plt.title('链路带宽与资源分配', fontsize=14)

# 第一个Y轴 - 带宽
ln1 = fig1.plot(time_points, bandwidth_link1, label='链路1带宽', color='blue')
ln2 = fig1.plot(time_points, bandwidth_link2, label='链路2带宽', color='green')
fig1.set_xlabel('时间(时隙)', fontsize=12)
fig1.set_ylabel('带宽(Mbps)', fontsize=12, color='blue')
fig1.tick_params(axis='y', labelcolor='blue')

# 添加探测标记 - 减少密度
probe_times1 = np.where(probe_link1)[0]
probe_times2 = np.where(probe_link2)[0]


# 第二个Y轴 - 数据包分配
ax2 = fig1.twinx()
ln3 = ax2.plot(time_points, w1_allocation, label='链路1分配(w1)', color='blue', linestyle='--')
ln4 = ax2.plot(time_points, w2_allocation, label='链路2分配(w2)', color='green', linestyle='--')
ax2.set_ylabel('数据包数量', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')

# 合并图例
lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
fig1.legend(lns, labs, loc='upper right')
fig1.grid(True)

# 图2：链路丢包率和冗余 - 使用双Y轴
plt.subplot(3, 1, 2)
fig2 = plt.gca()
plt.title('链路丢包率与冗余包数量', fontsize=14)

# 第一个Y轴 - 丢包率
ln1 = fig2.plot(time_points, packet_loss_link1, label='链路1丢包率', color='red')
ln2 = fig2.plot(time_points, packet_loss_link2, label='链路2丢包率', color='orange')
fig2.set_xlabel('时间(时隙)', fontsize=12)
fig2.set_ylabel('丢包率', fontsize=12, color='red')
fig2.tick_params(axis='y', labelcolor='red')

# 第二个Y轴 - 冗余包数量
ax2 = fig2.twinx()
ln3 = ax2.plot(time_points, redundancy, label='冗余包数量(R)', color='purple', linewidth=2)
ax2.set_ylabel('冗余包数量', fontsize=12, color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# 合并图例
lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
fig2.legend(lns, labs, loc='upper right')
fig2.grid(True)

# 图3：原始数据包和冗余数据包
plt.subplot(3, 1, 3)
plt.title('原始数据包和冗余数据包数量', fontsize=14)
plt.plot(time_points, original_packets, label='原始数据包(N)', color='blue', linewidth=2)
plt.plot(time_points, redundancy, label='冗余数据包(R)', color='red', linewidth=2)
plt.plot(time_points, w1_allocation + w2_allocation - redundancy, label='辅助1', color='green', linewidth=2)
plt.plot(time_points, redundancy, label='辅助2', color='purple', linewidth=2)


plt.xlabel('时间(时隙)', fontsize=12)
plt.ylabel('数据包数量', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)



plt.tight_layout()
plt.savefig('network_allocation_visualization.png', dpi=300)
plt.show()
