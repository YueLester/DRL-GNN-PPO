"""
海上浮标节点通信信道模拟
- 模拟两个海上浮标节点随洋流、风场和波浪因素移动
- 计算节点之间的信道状态(带宽、误码率、丢包率)随时间变化
- 总模拟时间：30分钟，有明显的信道变化
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

# 时间参数 - 修改为30分钟
total_time = 30 * 60  # 总模拟时间(秒)，模拟30分钟
dt = 5  # 时间步长(秒)，每5秒采样一次，增加采样密度
time_steps = int(total_time / dt)
time_minutes = np.linspace(0, 30, time_steps)  # 分钟为单位的时间轴


# 定义海洋环境参数
class OceanEnvironment:
    def __init__(self):
        # 洋流速度 (m/s) - 平均值和随时间变化的振幅
        self.current_mean = np.array([0.3, 0.25, 0])
        # 增加洋流变化频率，使其在30分钟内有明显变化
        self.current_amplitude = np.array([0.2, 0.2, 0])
        self.current_period = 15 * 60  # 洋流周期缩短为15分钟

        # 风速 (m/s) - 增加风速变化频率
        self.wind_mean = np.array([4.5, 3.0, 0])
        self.wind_amplitude = np.array([4.0, 3.0, 0])
        self.wind_period = 10 * 60  # 风向变化周期10分钟

        # 波浪参数 - 增加波浪变化频率
        self.wave_height_mean = 2.5
        self.wave_period_mean = 8.0
        self.wave_direction = np.array([1.0, 0.8, 0])
        self.wave_direction = self.wave_direction / np.linalg.norm(self.wave_direction)

        # 随机天气事件 - 增加暴风概率和减少持续时间，确保30分钟内出现
        self.storm_probability = 0.1  # 每个时间步有10%的概率出现暴风天气
        self.storm_duration = int(5 * 60 / dt)  # 暴风持续时间约5分钟
        self.storm_active = False
        self.storm_timer = 0

        # 添加快速变化的海况因素（如突发性涌浪）
        self.surge_active = False
        self.surge_probability = 0.05  # 5%概率出现涌浪
        self.surge_duration = int(2 * 60 / dt)  # 涌浪持续约2分钟
        self.surge_timer = 0

        # 确保在模拟开始后10分钟触发一次暴风
        self.scheduled_storm_time = int(10 * 60 / dt)
        self.scheduled_storm_triggered = False

    def get_current(self, t, position):
        # 基本洋流计算
        phase = 2 * np.pi * t / self.current_period
        temporal_variation = self.current_amplitude * np.sin(phase)

        # 增加空间变异性
        spatial_variation = np.array([
            0.15 * np.sin(0.005 * position[0] + 0.008 * position[1]),
            0.15 * np.cos(0.005 * position[0] + 0.008 * position[1]),
            0
        ])

        # 添加高频变化
        high_freq_variation = 0.1 * np.array([
            np.sin(2 * np.pi * t / 60),
            np.cos(2 * np.pi * t / 45),
            0
        ])

        # 随机扰动
        random_disturbance = 0.1 * np.random.randn(3)
        random_disturbance[2] = 0  # 保持z方向为零

        # 暴风影响
        storm_factor = 1.0
        if self.storm_active:
            storm_factor = 3.0  # 暴风时洋流增强3倍

        # 涌浪影响
        surge_factor = 1.0
        if self.surge_active:
            surge_factor = 2.0  # 涌浪时局部增强2倍

        return (self.current_mean + temporal_variation + spatial_variation +
                high_freq_variation + random_disturbance) * storm_factor * surge_factor

    def get_wind(self, t):
        # 基本风场
        phase = 2 * np.pi * t / self.wind_period
        base_wind = self.wind_mean + self.wind_amplitude * np.sin(phase)

        # 添加高频变化以模拟阵风
        gust_effect = 2.0 * np.array([
            np.sin(2 * np.pi * t / 30),
            np.cos(2 * np.pi * t / 20),
            0
        ])

        # 突变模拟 - 每3分钟可能出现一次方向突变
        direction_shift = np.array([0, 0, 0])
        if t % (3 * 60) < 30:  # 在每3分钟的开始30秒内
            shift_factor = np.sin(2 * np.pi * t / 30)
            direction_shift = np.array([shift_factor, -shift_factor, 0]) * 2.0

        # 暴风和涌浪影响
        storm_factor = 1.0
        if self.storm_active:
            storm_factor = 3.5  # 暴风时风速增强3.5倍

        return (base_wind + gust_effect + direction_shift) * storm_factor

    def get_wave_effect(self, t, position):
        # 短周期波浪变化，使其在30分钟内更明显
        short_term = 0.5 * np.sin(2 * np.pi * t / (2 * 60))  # 2分钟周期
        medium_term = 0.3 * np.sin(2 * np.pi * t / (5 * 60))  # 5分钟周期
        long_term = 0.2 * np.sin(2 * np.pi * t / (15 * 60))  # 15分钟周期

        wave_height = self.wave_height_mean * (1 + short_term + medium_term + long_term)

        # 暴风影响
        if self.storm_active:
            wave_height *= 2.5  # 暴风时波高增加2.5倍

        # 涌浪影响
        if self.surge_active:
            wave_height *= 1.8  # 涌浪时波高增加1.8倍

        # 增强斯托克斯漂移效应
        wave_effect = 0.06 * wave_height ** 2 * self.wave_direction

        # 随机扰动 - 增强随机性
        random_disturb = 0.04 * np.random.randn(3) * wave_height
        random_disturb[2] = 0  # 保持z方向为零

        # 周期性的群波效应
        if (t % (4 * 60)) < 60:  # 每4分钟出现一次持续1分钟的群波
            group_wave = 0.15 * wave_height * self.wave_direction
        else:
            group_wave = np.zeros(3)

        return wave_effect + random_disturb + group_wave

    def update_weather(self, t, time_step):
        # 检查是否应触发计划的暴风
        if not self.scheduled_storm_triggered and time_step >= self.scheduled_storm_time:
            self.storm_active = True
            self.storm_timer = self.storm_duration
            self.scheduled_storm_triggered = True

        # 随机暴风更新
        elif not self.storm_active and np.random.random() < self.storm_probability:
            self.storm_active = True
            self.storm_timer = self.storm_duration

        # 暴风倒计时
        if self.storm_active:
            self.storm_timer -= 1
            if self.storm_timer <= 0:
                self.storm_active = False

        # 涌浪更新
        if not self.surge_active and np.random.random() < self.surge_probability:
            self.surge_active = True
            self.surge_timer = self.surge_duration

        # 涌浪倒计时
        if self.surge_active:
            self.surge_timer -= 1
            if self.surge_timer <= 0:
                self.surge_active = False


# 浮标节点类 - 增强随机移动和环境响应
class FloatingNode:
    def __init__(self, initial_position, drag_coefficient=0.9, wind_effect_factor=0.05):
        self.position = np.array(initial_position)
        self.velocity = np.zeros(3)
        self.drag_coefficient = drag_coefficient  # 增加拖曳系数
        self.wind_effect_factor = wind_effect_factor  # 增加风的影响因子
        self.trajectory = [initial_position]
        self.random_movement_factor = 0.05  # 增加随机移动因子
        self.last_direction_change = 0

    def update(self, env, t, dt, time_step):
        # 计算各种作用力产生的速度
        current_velocity = env.get_current(t, self.position)
        wind_velocity = env.get_wind(t) * self.wind_effect_factor
        wave_velocity = env.get_wave_effect(t, self.position)

        # 随机方向变化 - 每1分钟可能改变一次随机移动方向
        if time_step - self.last_direction_change >= int(60 / dt):
            self.random_direction = np.random.randn(3)
            self.random_direction[2] = 0  # 保持z为零
            self.random_direction = self.random_direction / max(0.001, np.linalg.norm(self.random_direction))
            self.last_direction_change = time_step

        # 随机移动组件
        random_velocity = self.random_movement_factor * self.random_direction

        # 增加环境条件的影响因素
        if env.storm_active:
            random_velocity *= 3.0
            self.drag_coefficient = min(0.95, self.drag_coefficient + 0.01)  # 暴风中阻力增加
        else:
            self.drag_coefficient = max(0.85, self.drag_coefficient - 0.001)  # 正常情况下阻力恢复

        if env.surge_active:
            random_velocity *= 2.0

        # 总速度
        self.velocity = self.drag_coefficient * (current_velocity + wind_velocity + wave_velocity) + random_velocity

        # 速度突变 - 模拟碰到物体或异常水流
        if np.random.random() < 0.01:  # 1%的概率
            self.velocity += np.random.randn(3) * 0.2
            self.velocity[2] = 0  # 保持z为零

        # 更新位置
        self.position = self.position + self.velocity * dt
        self.trajectory.append(self.position.copy())

        return self.position, self.velocity


# 信道状态计算函数 - 增强时变特性
def calculate_channel_state(node1, node2, env, t):
    # 计算节点间距离
    distance = np.linalg.norm(node1.position - node2.position)

    # 相对速度
    relative_velocity = np.linalg.norm(node1.velocity - node2.velocity)

    # 带宽计算 - 增加更多变化因素
    base_bandwidth = 5e6  # 5 MHz 基础带宽

    # 距离影响
    distance_factor = np.exp(-0.002 * distance)

    # 位置相关因子 - 模拟空间不均匀特性
    x_mean = (node1.position[0] + node2.position[0]) / 2
    y_mean = (node1.position[1] + node2.position[1]) / 2
    position_factor = 0.7 + 0.3 * np.sin(0.01 * x_mean) * np.cos(0.01 * y_mean)

    # 波高影响 - 30分钟内变化更明显
    wave_height = env.wave_height_mean * (1 + 0.5 * np.sin(2 * np.pi * t / (3 * 60)))
    if env.storm_active:
        wave_height *= 2.5
    if env.surge_active:
        wave_height *= 1.8

    sea_state_factor = np.exp(-0.5 * wave_height)

    # 快速小幅波动 - 增加变化频率
    fast_fluctuation = 0.3 * np.sin(2 * np.pi * t / 15) + 0.2 * np.cos(2 * np.pi * t / 10)

    # 带宽周期性跳变 - 模拟设备切换信道或干扰
    periodic_jump = 1.0
    if (t % 180) < 20:  # 每3分钟出现一次持续20秒的带宽跳变
        periodic_jump = 0.4 + 0.6 * np.random.random()

    # 综合计算带宽
    bandwidth = base_bandwidth * distance_factor * sea_state_factor * position_factor * (
                1 + 0.2 * fast_fluctuation) * periodic_jump

    # 异常情况 - 信道突然中断或恢复
    if np.random.random() < 0.005:  # 0.5%概率
        bandwidth *= 0.1  # 带宽突然下降到10%

    # 暴风和涌浪影响
    if env.storm_active:
        bandwidth *= 0.3  # 暴风时带宽降至30%
    if env.surge_active:
        bandwidth *= 0.6  # 涌浪时带宽降至60%

    # 误码率计算 - 增加时变特性
    base_ber = 1e-5
    ber_distance_factor = 1 + 0.05 * distance  # 距离影响更明显
    ber_velocity_factor = 1 + 1.2 * relative_velocity  # 速度影响更明显
    ber_sea_factor = 1 + 2.0 * wave_height  # 海况影响更明显

    # 快速波动 - 频率更高
    ber_fluctuation = 0.8 * np.sin(2 * np.pi * t / 8) + 0.6 * np.cos(2 * np.pi * t / 5)

    # 添加突发错误事件
    burst_error = 1.0
    if np.random.random() < 0.01:  # 1%概率出现突发错误
        burst_error = 10.0 + 20.0 * np.random.random()

    # 综合计算误码率
    ber = base_ber * ber_distance_factor * ber_velocity_factor * ber_sea_factor * (
                1 + 0.5 * ber_fluctuation) * burst_error

    # 异常情况 - 信道质量突变
    if (t % 240) < 30:  # 每4分钟出现一次持续30秒的周期性干扰
        ber *= (3.0 + 4.0 * np.random.random())

    # 暴风和涌浪影响
    if env.storm_active:
        ber *= 5.0  # 暴风时误码率增加5倍
    if env.surge_active:
        ber *= 2.5  # 涌浪时误码率增加2.5倍

    # 丢包率计算 - 增加变化和非线性关系
    base_plr = 0.005  # 增加基础丢包率

    # 与误码率的非线性关系
    plr_threshold = 0.00001
    if ber > plr_threshold:
        plr_ber_factor = 0.3 * (ber / plr_threshold) ** 2.0
    else:
        plr_ber_factor = 0.3

    # 波高影响
    plr_wave_factor = 0.15 * wave_height ** 1.5

    # 相对速度影响
    plr_velocity_factor = 0.1 * relative_velocity

    # 快速波动
    plr_fluctuation = 0.4 * np.sin(2 * np.pi * t / 12) + 0.3 * np.cos(2 * np.pi * t / 7)

    # 突发丢包事件
    burst_loss = 1.0
    if np.random.random() < 0.02:  # 2%概率
        burst_loss = 2.0 + 3.0 * np.random.random()

    # 综合计算丢包率
    plr = base_plr + plr_ber_factor + plr_wave_factor + plr_velocity_factor + 0.2 * plr_fluctuation
    plr *= burst_loss

    # 周期性拥塞 - 模拟网络负载波动
    if (t % 300) < 60:  # 每5分钟出现一次持续1分钟的拥塞
        plr += 0.2 * np.random.random()

    # 暴风和涌浪影响
    if env.storm_active:
        plr = min(0.9, plr * 3.0)  # 暴风时丢包率增加，但最大不超过90%
    if env.surge_active:
        plr = min(0.7, plr * 1.8)  # 涌浪时丢包率增加，但最大不超过70%
    else:
        plr = min(plr, 1.0)  # 确保丢包率不超过1

    return bandwidth, ber, plr


# 可视化函数
def visualize_results(time, positions1, positions2, bandwidths, bers, plrs, distances, storm_periods, surge_periods):
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
    ax1.set_title('节点轨迹 (30分钟模拟)')
    ax1.legend()

    # 节点间距离
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, distances, 'k-')
    # 标记暴风和涌浪区域
    for i in range(len(time) - 1):
        if storm_periods[i] > 0:
            ax2.axvspan(time[i], time[i + 1], alpha=0.3, color='red')
        if surge_periods[i] > 0:
            ax2.axvspan(time[i], time[i + 1], alpha=0.2, color='orange')
    ax2.set_xlabel('时间 (分钟)')
    ax2.set_ylabel('距离 (m)')
    ax2.set_title('节点间距离随时间变化')
    ax2.grid(True)

    # 带宽
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, bandwidths / 1e6, 'g-')
    # 标记暴风和涌浪区域
    for i in range(len(time) - 1):
        if storm_periods[i] > 0:
            ax3.axvspan(time[i], time[i + 1], alpha=0.3, color='red')
        if surge_periods[i] > 0:
            ax3.axvspan(time[i], time[i + 1], alpha=0.2, color='orange')
    ax3.set_xlabel('时间 (分钟)')
    ax3.set_ylabel('带宽 (MHz)')
    ax3.set_title('信道带宽随时间变化')
    ax3.grid(True)

    # 误码率
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.semilogy(time, bers, 'r-')
    # 标记暴风和涌浪区域
    for i in range(len(time) - 1):
        if storm_periods[i] > 0:
            ax4.axvspan(time[i], time[i + 1], alpha=0.3, color='red')
        if surge_periods[i] > 0:
            ax4.axvspan(time[i], time[i + 1], alpha=0.2, color='orange')
    ax4.set_xlabel('时间 (分钟)')
    ax4.set_ylabel('误码率')
    ax4.set_title('误码率随时间变化')
    ax4.grid(True)

    # 丢包率
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time, plrs, 'm-')
    # 标记暴风和涌浪区域
    for i in range(len(time) - 1):
        if storm_periods[i] > 0:
            ax5.axvspan(time[i], time[i + 1], alpha=0.3, color='red')
        if surge_periods[i] > 0:
            ax5.axvspan(time[i], time[i + 1], alpha=0.2, color='orange')
    ax5.set_xlabel('时间 (分钟)')
    ax5.set_ylabel('丢包率')
    ax5.set_title('丢包率随时间变化')
    ax5.grid(True)

    # 增加图例说明
    fig.text(0.15, 0.01, '红色区域: 暴风状态', color='red', fontsize=10)
    fig.text(0.35, 0.01, '橙色区域: 涌浪状态', color='orange', fontsize=10)

    plt.tight_layout()
    plt.savefig('channel_state_simulation_30min.png', dpi=300)
    plt.show()


# 主模拟函数
def run_simulation():
    # 创建海洋环境
    env = OceanEnvironment()

    # 创建两个浮标节点 - 初始位置更近
    node1 = FloatingNode([0, 0, 0])
    node2 = FloatingNode([200, 100, 0])

    # 初始化随机方向
    node1.random_direction = np.array([1.0, 0.5, 0])
    node1.random_direction = node1.random_direction / np.linalg.norm(node1.random_direction)
    node2.random_direction = np.array([-0.5, 1.0, 0])
    node2.random_direction = node2.random_direction / np.linalg.norm(node2.random_direction)

    # 存储结果的数组
    positions1 = np.zeros((time_steps, 3))
    positions2 = np.zeros((time_steps, 3))
    bandwidths = np.zeros(time_steps)
    bers = np.zeros(time_steps)
    plrs = np.zeros(time_steps)
    distances = np.zeros(time_steps)
    storm_periods = np.zeros(time_steps)  # 记录暴风状态
    surge_periods = np.zeros(time_steps)  # 记录涌浪状态

    # 主模拟循环
    for i in range(time_steps):
        t = i * dt

        # 更新天气状态
        env.update_weather(t, i)
        storm_periods[i] = 1 if env.storm_active else 0
        surge_periods[i] = 1 if env.surge_active else 0

        # 更新节点位置
        pos1, vel1 = node1.update(env, t, dt, i)
        pos2, vel2 = node2.update(env, t, dt, i)

        positions1[i] = pos1
        positions2[i] = pos2

        # 计算信道状态
        bandwidth, ber, plr = calculate_channel_state(node1, node2, env, t)

        bandwidths[i] = bandwidth
        bers[i] = ber
        plrs[i] = plr
        distances[i] = np.linalg.norm(pos1 - pos2)

    # 可视化结果，添加暴风和涌浪区域标记
    visualize_results(time_minutes, positions1, positions2, bandwidths, bers, plrs, distances, storm_periods,
                      surge_periods)


# 程序入口
if __name__ == "__main__":
    print("开始海上浮标通信信道模拟 (30分钟)...")
    run_simulation()
    print("模拟完成，结果已显示。")
