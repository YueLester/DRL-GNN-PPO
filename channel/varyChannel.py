
"""
海上浮标节点通信信道模拟
- 模拟两个海上浮标节点随洋流、风场和波浪因素移动
- 计算节点之间的信道状态(带宽、误码率、丢包率)随时间变化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题



# 设置随机种子以确保结果可重现
np.random.seed(42)

# 时间参数
total_time = 24 * 3600  # 总模拟时间(秒)，模拟24小时
dt = 300  # 时间步长(秒)，每5分钟
time_steps = int(total_time / dt)
time_hours = np.linspace(0, 24, time_steps)  # 小时为单位的时间轴


# 定义海洋环境参数
class OceanEnvironment:
    def __init__(self):
        # 洋流速度 (m/s) - 平均值和随时间变化的振幅
        self.current_mean = np.array([0.3, 0.25, 0])  # 增加平均洋流速度
        self.current_amplitude = np.array([0.15, 0.15, 0])  # 增大振幅，从0.05增至0.15
        self.current_period = 8 * 3600  # 缩短洋流周期，从12小时减至8小时

        # 风速 (m/s)
        self.wind_mean = np.array([4.5, 3.0, 0])  # 增加平均风速
        self.wind_amplitude = np.array([3.0, 2.5, 0])  # 增大风速振幅，从1.5/1.0增至3.0/2.5
        self.wind_period = 4 * 3600  # 缩短风向变化周期，从6小时减至4小时

        # 波浪参数
        self.wave_height_mean = 2.5  # 增加平均波高，从1.0米增至2.5米
        self.wave_period_mean = 8.0  # 缩短波浪周期，从10秒减至8秒
        self.wave_direction = np.array([1.0, 0.8, 0])  # 调整波浪方向
        self.wave_direction = self.wave_direction / np.linalg.norm(self.wave_direction)

        # 添加随机天气事件
        self.storm_probability = 0.05  # 每个时间步有5%的概率出现暴风天气
        self.storm_duration = int(2 * 3600 / 300)  # 暴风持续时间约2小时
        self.storm_active = False
        self.storm_timer = 0

    # 修改洋流计算方法，增加随机扰动
    def get_current(self, t, position):
        # 计算基本洋流
        phase = 2 * np.pi * t / self.current_period
        temporal_variation = self.current_amplitude * np.sin(phase)

        # 增强空间变化
        spatial_variation = np.array([
            0.12 * np.sin(0.002 * position[0] + 0.003 * position[1]),
            0.12 * np.cos(0.002 * position[0] + 0.003 * position[1]),
            0
        ])

        # 添加随机扰动
        random_disturbance = 0.08 * np.random.randn(3)
        random_disturbance[2] = 0  # 保持z方向为零

        # 检查是否处于暴风状态
        storm_factor = 1.0
        if self.storm_active:
            storm_factor = 2.5  # 暴风时洋流增强2.5倍

        return (self.current_mean + temporal_variation + spatial_variation + random_disturbance) * storm_factor

    # 修改风场计算方法，增加突变
    def get_wind(self, t):
        # 基本风场
        phase = 2 * np.pi * t / self.wind_period
        base_wind = self.wind_mean + self.wind_amplitude * np.sin(phase)

        # 添加短期波动
        short_variation = 0.8 * np.array([
            np.sin(2 * np.pi * t / (0.5 * 3600)),
            np.cos(2 * np.pi * t / (0.7 * 3600)),
            0
        ])

        # 检查是否处于暴风状态
        storm_factor = 1.0
        if self.storm_active:
            storm_factor = 3.0  # 暴风时风速增强3倍

        return (base_wind + short_variation) * storm_factor

    # 修改波浪效应计算，增加复杂度
    def get_wave_effect(self, t, position):
        # 波高随时间变化，增加短周期和长周期振荡
        short_term = 0.3 * np.sin(2 * np.pi * t / (1 * 3600))  # 1小时周期
        long_term = 0.4 * np.sin(2 * np.pi * t / (12 * 3600))  # 12小时周期
        wave_height = self.wave_height_mean * (1 + short_term + long_term)

        # 如果处于暴风状态，波高增加
        if self.storm_active:
            wave_height *= 2.2  # 暴风时波高增加2.2倍

        # 增强斯托克斯漂移效应
        wave_effect = 0.04 * wave_height ** 2 * self.wave_direction

        # 增强随机扰动
        random_disturb = 0.025 * np.random.randn(3) * wave_height
        random_disturb[2] = 0  # 保持z方向为零

        return wave_effect + random_disturb

    # 添加暴风天气状态更新
    def update_weather(self):
        # 如果当前不在暴风中，有一定概率触发暴风
        if not self.storm_active and np.random.random() < self.storm_probability:
            self.storm_active = True
            self.storm_timer = self.storm_duration

        # 如果正在暴风中，倒计时
        if self.storm_active:
            self.storm_timer -= 1
            if self.storm_timer <= 0:
                self.storm_active = False


# 修改浮标节点类增加随机移动
class FloatingNode:
    def __init__(self, initial_position, drag_coefficient=0.85, wind_effect_factor=0.04):
        self.position = np.array(initial_position)
        self.velocity = np.zeros(3)
        self.drag_coefficient = drag_coefficient  # 增加拖曳系数
        self.wind_effect_factor = wind_effect_factor  # 增加风的影响因子到4%
        self.trajectory = [initial_position]
        self.random_movement_factor = 0.02  # 添加随机移动因子

    def update(self, env, t, dt):
        # 计算各种作用力产生的速度
        current_velocity = env.get_current(t, self.position)
        wind_velocity = env.get_wind(t) * self.wind_effect_factor
        wave_velocity = env.get_wave_effect(t, self.position)

        # 添加随机移动
        random_velocity = self.random_movement_factor * np.random.randn(3)
        random_velocity[2] = 0  # 保持z为零

        # 如果处于暴风状态，增加随机性
        if env.storm_active:
            random_velocity *= 2.5

        # 总速度
        self.velocity = self.drag_coefficient * (current_velocity + wind_velocity + wave_velocity) + random_velocity

        # 更新位置
        self.position = self.position + self.velocity * dt
        self.trajectory.append(self.position.copy())

        return self.position, self.velocity


# 修改信道状态计算函数
def calculate_channel_state(node1, node2, env, t):
    # 计算节点间距离
    distance = np.linalg.norm(node1.position - node2.position)

    # 相对速度
    relative_velocity = np.linalg.norm(node1.velocity - node2.velocity)

    # 计算时变带宽 (增加变化幅度)
    base_bandwidth = 5e6  # 5 MHz 基础带宽

    # 距离衰减更剧烈
    distance_factor = np.exp(-0.001 * distance)  # 修改系数使衰减更敏感

    # 波高影响更显著
    wave_height = env.wave_height_mean * (1 + 0.3 * np.sin(2 * np.pi * t / (8 * 3600)))
    if env.storm_active:
        wave_height *= 2.0  # 暴风期间波高加倍

    sea_state_factor = np.exp(-0.4 * wave_height)  # 增加海况影响因子

    # 添加快速小幅波动
    fast_fluctuation = 0.3 * np.sin(2 * np.pi * t / (300)) + 0.2 * np.cos(2 * np.pi * t / (180))

    # 综合计算带宽
    bandwidth = base_bandwidth * distance_factor * sea_state_factor * (1 + 0.1 * fast_fluctuation)

    # 暴风影响
    if env.storm_active:
        bandwidth *= 0.4  # 暴风时带宽降至40%

    # 误码率计算 (增加波动)
    base_ber = 1e-5  # 基础误码率
    ber_distance_factor = 1 + 0.02 * distance  # 增加距离敏感度
    ber_velocity_factor = 1 + 0.8 * relative_velocity  # 增加相对速度影响
    ber_sea_factor = 1 + 1.5 * wave_height  # 增加海况影响

    # 添加快速波动
    ber_fluctuation = 0.5 * np.sin(2 * np.pi * t / (120)) + 0.3 * np.cos(2 * np.pi * t / (240))

    # 综合计算误码率
    ber = base_ber * ber_distance_factor * ber_velocity_factor * ber_sea_factor * (1 + 0.2 * ber_fluctuation)

    # 暴风影响
    if env.storm_active:
        ber *= 3.0  # 暴风时误码率增加3倍

    # 丢包率计算 (增加变化)
    base_plr = 0.002  # 增加基础丢包率

    # 增加非线性关系
    plr_threshold = 0.00001
    if ber > plr_threshold:
        plr_ber_factor = 0.2 * (ber / plr_threshold) ** 1.5
    else:
        plr_ber_factor = 0.2

    # 波高影响更显著
    plr_wave_factor = 0.1 * wave_height ** 1.2

    # 添加快速波动
    plr_fluctuation = 0.3 * np.sin(2 * np.pi * t / (180)) + 0.2 * np.cos(2 * np.pi * t / (90))

    # 综合计算丢包率
    plr = base_plr + plr_ber_factor + plr_wave_factor + 0.1 * plr_fluctuation

    # 暴风影响
    if env.storm_active:
        plr = min(0.8, plr * 2.5)  # 暴风时丢包率增加，但最大不超过80%
    else:
        plr = min(plr, 1.0)  # 确保丢包率不超过1

    return bandwidth, ber, plr


# 可视化函数
def visualize_results(time, positions1, positions2, bandwidths, bers, plrs, distances, storm_periods):
    # 创建一个包含多个子图的大图
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])

    # 3D轨迹图
    ax1 = fig.add_subplot(gs[0, :], projection='3d')
    ax1.plot(positions1[:, 0], positions1[:, 1], positions1[:, 2], 'b-', label='Node 1')
    ax1.plot(positions2[:, 0], positions2[:, 1], positions2[:, 2], 'r-', label='Node 2')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('节点轨迹')
    ax1.legend()

    # 节点间距离
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, distances, 'k-')
    # 标记暴风区域
    for i in range(len(time) - 1):
        if storm_periods[i] > 0:
            ax2.axvspan(time[i], time[i + 1], alpha=0.3, color='red')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('距离 (m)')
    ax2.set_title('节点间距离随时间变化')
    ax2.grid(True)

    # 带宽
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, bandwidths / 1e6, 'g-')
    # 标记暴风区域
    for i in range(len(time) - 1):
        if storm_periods[i] > 0:
            ax3.axvspan(time[i], time[i + 1], alpha=0.3, color='red')
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('带宽 (MHz)')
    ax3.set_title('信道带宽随时间变化')
    ax3.grid(True)

    # 误码率
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.semilogy(time, bers, 'r-')
    # 标记暴风区域
    for i in range(len(time) - 1):
        if storm_periods[i] > 0:
            ax4.axvspan(time[i], time[i + 1], alpha=0.3, color='red')
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('误码率')
    ax4.set_title('误码率随时间变化')
    ax4.grid(True)

    # 丢包率
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time, plrs, 'm-')
    # 标记暴风区域
    for i in range(len(time) - 1):
        if storm_periods[i] > 0:
            ax5.axvspan(time[i], time[i + 1], alpha=0.3, color='red')
    ax5.set_xlabel('时间 (小时)')
    ax5.set_ylabel('丢包率')
    ax5.set_title('丢包率随时间变化')
    ax5.grid(True)

    plt.tight_layout()
    plt.savefig('channel_state_simulation.png', dpi=300)
    plt.show()


# 主模拟函数
def run_simulation():
    # 创建海洋环境
    env = OceanEnvironment()

    # 创建两个浮标节点
    node1 = FloatingNode([0, 0, 0])
    node2 = FloatingNode([500, 200, 0])

    # 存储结果的数组
    positions1 = np.zeros((time_steps, 3))
    positions2 = np.zeros((time_steps, 3))
    bandwidths = np.zeros(time_steps)
    bers = np.zeros(time_steps)
    plrs = np.zeros(time_steps)
    distances = np.zeros(time_steps)
    storm_periods = np.zeros(time_steps)  # 记录暴风状态

    # 主模拟循环
    for i in range(time_steps):
        t = i * dt

        # 更新天气状态
        env.update_weather()
        storm_periods[i] = 1 if env.storm_active else 0

        # 更新节点位置
        pos1, vel1 = node1.update(env, t, dt)
        pos2, vel2 = node2.update(env, t, dt)

        positions1[i] = pos1
        positions2[i] = pos2

        # 计算信道状态
        bandwidth, ber, plr = calculate_channel_state(node1, node2, env, t)

        bandwidths[i] = bandwidth
        bers[i] = ber
        plrs[i] = plr
        distances[i] = np.linalg.norm(pos1 - pos2)

    # 可视化结果，添加暴风区域标记
    visualize_results(time_hours, positions1, positions2, bandwidths, bers, plrs, distances, storm_periods)


# 程序入口
if __name__ == "__main__":
    print("开始海上浮标通信信道模拟...")
    run_simulation()
    print("模拟完成，结果已显示。")
