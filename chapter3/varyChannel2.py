"""
海上浮标节点通信信道模拟
- 模拟两个海上浮标节点随洋流、风场和波浪因素移动
- 计算节点之间的信道状态(带宽、误码率、丢包率)随时间变化
- 单独展示四张图表
- 仿真时长为30分钟
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# 设置随机种子以确保结果可重现
np.random.seed(63)

# 时间参数 - 修改为30分钟
total_time = 30 * 60  # 总模拟时间(秒)，模拟30分钟
dt = 10  # 时间步长(秒)，缩短到10秒以获得更多数据点
time_steps = int(total_time / dt)
time_minutes = np.linspace(0, 30, time_steps)  # 分钟为单位的时间轴


# 定义海洋环境参数
class OceanEnvironment:
    def __init__(self):
        # 缩短时间周期以适应30分钟的模拟
        # 洋流速度 (m/s) - 平均值和随时间变化的振幅
        self.current_mean = np.array([0.3, 0.25, 0])
        self.current_amplitude = np.array([0.15, 0.15, 0])
        self.current_period = 15 * 60  # 洋流周期缩短到15分钟

        # 风速 (m/s)
        self.wind_mean = np.array([4.5, 3.0, 0])
        self.wind_amplitude = np.array([3.0, 2.5, 0])
        self.wind_period = 10 * 60  # 风向变化周期缩短到10分钟

        # 波浪参数
        self.wave_height_mean = 2.5
        self.wave_period_mean = 8.0
        self.wave_direction = np.array([1.0, 0.8, 0])
        self.wave_direction = self.wave_direction / np.linalg.norm(self.wave_direction)

        # 添加随机天气事件
        self.storm_probability = 0.03  # 每个时间步有3%的概率出现暴风天气
        self.storm_duration = int(3 * 60 / dt)  # 暴风持续时间约3分钟
        self.storm_active = False
        self.storm_timer = 0

        # 跟踪波浪强度历史
        self.wave_heights = []

        # 记录暴风开始时间点，用于标记图表
        self.storm_start_times = []
        self.storm_end_times = []  # 添加结束时间记录

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

    # 修改风场计算方法，增加更多频率的波动
    def get_wind(self, t):
        # 基本风场
        phase = 2 * np.pi * t / self.wind_period
        base_wind = self.wind_mean + self.wind_amplitude * np.sin(phase)

        # 添加多频率波动
        short_variation = np.array([
            0.6 * np.sin(2 * np.pi * t / (60)) + 0.4 * np.sin(2 * np.pi * t / (30)),
            0.5 * np.cos(2 * np.pi * t / (45)) + 0.3 * np.cos(2 * np.pi * t / (20)),
            0
        ])

        # 检查是否处于暴风状态
        storm_factor = 1.0
        if self.storm_active:
            storm_factor = 3.0  # 暴风时风速增强3倍

        return (base_wind + short_variation) * storm_factor

    # 修改波浪效应计算，增加复杂度
    def get_wave_effect(self, t, position):
        # 波高随时间变化，增加多频率振荡
        short_term = 0.3 * np.sin(2 * np.pi * t / (2 * 60))  # 2分钟周期
        medium_term = 0.25 * np.sin(2 * np.pi * t / (5 * 60))  # 5分钟周期
        long_term = 0.2 * np.sin(2 * np.pi * t / (15 * 60))  # 15分钟周期

        # 增加随机噪声
        noise = 0.1 * np.random.randn()

        wave_height = self.wave_height_mean * (1 + short_term + medium_term + long_term + noise)

        # 如果处于暴风状态，波高增加
        if self.storm_active:
            wave_height *= 2.2  # 暴风时波高增加2.2倍

        # 保存当前波高
        self.wave_heights.append(wave_height)

        # 增强斯托克斯漂移效应
        wave_effect = 0.04 * wave_height ** 2 * self.wave_direction

        # 增强随机扰动
        random_disturb = 0.025 * np.random.randn(3) * wave_height
        random_disturb[2] = 0  # 保持z方向为零

        return wave_effect + random_disturb, wave_height

    # 添加暴风天气状态更新
    def update_weather(self, t):
        # 如果当前不在暴风中，有一定概率触发暴风
        if not self.storm_active and np.random.random() < self.storm_probability:
            self.storm_active = True
            self.storm_timer = self.storm_duration
            # 记录暴风开始时间
            start_time = t/60  # 转换为分钟
            self.storm_start_times.append(start_time)
            # 计算结束时间
            end_time = start_time + (self.storm_duration * dt / 60)  # 转换为分钟
            self.storm_end_times.append(end_time)

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
        wave_effect, _ = env.get_wave_effect(t, self.position)

        # 添加随机移动
        random_velocity = self.random_movement_factor * np.random.randn(3)
        random_velocity[2] = 0  # 保持z为零

        # 如果处于暴风状态，增加随机性
        if env.storm_active:
            random_velocity *= 2.5

        # 总速度
        self.velocity = self.drag_coefficient * (current_velocity + wind_velocity + wave_effect) + random_velocity

        # 更新位置
        self.position = self.position + self.velocity * dt
        self.trajectory.append(self.position.copy())

        return self.position, self.velocity


# 修改信道状态计算函数，特别是丢包率计算，避免丢包率持续为1
# 修改信道状态计算函数，特别是误码率计算部分，使其每5分钟呈现单调趋势但带有震荡
def calculate_channel_state(node1, node2, env, t, wave_height):
    # 计算节点间距离
    distance = np.linalg.norm(node1.position - node2.position)

    # 相对速度
    relative_velocity = np.linalg.norm(node1.velocity - node2.velocity)

    # 计算时变带宽 (增加变化幅度)
    base_bandwidth = 5e6  # 5 MHz 基础带宽

    # 距离衰减更剧烈
    distance_factor = np.exp(-0.001 * distance)  # 修改系数使衰减更敏感

    # 波高影响更显著
    sea_state_factor = np.exp(-0.4 * wave_height)  # 增加海况影响因子

    # 添加多频率波动
    fast_fluctuation = (
            0.2 * np.sin(2 * np.pi * t / (30)) +
            0.15 * np.cos(2 * np.pi * t / (15)) +
            0.1 * np.sin(2 * np.pi * t / (7))
    )

    # 综合计算带宽
    bandwidth = base_bandwidth * distance_factor * sea_state_factor * (1 + fast_fluctuation)

    # 暴风影响
    if env.storm_active:
        bandwidth *= 0.4  # 暴风时带宽降至40%

    # 修改误码率计算，使其每5分钟呈现单调但有震荡的特点

    # 首先确定当前所处的5分钟周期(0-5, 5-10, 10-15, 15-20, 20-25, 25-30)
    period_index = int(t / (5 * 60))  # 每5分钟为一个周期
    time_in_period = t % (5 * 60)  # 当前在周期内的时间点
    period_progress = time_in_period / (5 * 60)  # 周期内的进度比例(0-1)

    # 各周期的误码率基础值和趋势
    # 奇数周期误码率趋势递增，偶数周期误码率趋势递减
    if period_index % 2 == 0:  # 偶数周期(0,2,4) - 递减趋势
        period_trend = 1.0 - 0.7 * period_progress  # 从1.0递减到0.3
    else:  # 奇数周期(1,3,5) - 递增趋势
        period_trend = 0.3 + 0.7 * period_progress  # 从0.3递增到1.0

    # 基础误码率 - 先确定各周期基准值
    period_base_values = [1e-5, 5e-5, 2e-5, 8e-5, 3e-5, 6e-5]  # 为6个周期设置不同的基准值
    current_base_ber = period_base_values[min(period_index, 5)]  # 防止索引越界

    # 应用周期内的趋势系数
    base_ber = current_base_ber * period_trend

    # 添加高频震荡 - 使用多个不同频率的正弦波叠加
    # 频率选择使其足够高以在5分钟周期内有多次震荡
    ber_oscillation = (
            0.25 * np.sin(2 * np.pi * t / (30)) +  # 30秒周期
            0.15 * np.sin(2 * np.pi * t / (15)) +  # 15秒周期
            0.1 * np.sin(2 * np.pi * t / (7))  # 7秒周期
    )

    # 距离和海况影响因子保持不变
    ber_distance_factor = 1 + 0.02 * distance
    ber_velocity_factor = 1 + 0.8 * relative_velocity
    ber_sea_factor = 1 + 1.5 * wave_height

    # 综合计算误码率 - 基础趋势加上震荡
    ber = base_ber * ber_distance_factor * ber_velocity_factor * ber_sea_factor * (1 + ber_oscillation)

    # 暴风影响
    if env.storm_active:
        ber *= 3.0  # 暴风时误码率增加3倍

    # 剩余的丢包率计算保持不变
    # ... existing code ...

    # 基础丢包率 - 较低值开始
    base_plr = 0.001

    # 距离对丢包的影响 - 使用指数函数避免线性增长过快
    distance_norm = distance / 1000  # 归一化距离
    plr_distance = 0.02 * (1 - np.exp(-distance_norm))

    # 波高对丢包的影响 - 使用平方根函数减缓增长速度
    wave_norm = wave_height / 5  # 归一化波高
    plr_wave = 0.03 * np.sqrt(wave_norm)

    # 误码率对丢包的影响 - 避免过大比例
    ber_norm = min(ber / 1e-3, 1.0)  # 将ber归一化，避免过大值
    plr_ber = 0.05 * ber_norm

    # 多频率波动 - 使用更多不同周期的波动
    plr_oscillation = (
            0.01 * np.sin(2 * np.pi * t / (150)) +
            0.008 * np.cos(2 * np.pi * t / (90)) +
            0.006 * np.sin(2 * np.pi * t / (45)) +
            0.004 * np.cos(2 * np.pi * t / (20))
    )

    # 随机波动因子 - 添加较小的随机性，避免过大波动
    random_factor = 0.005 * np.random.randn()

    # 综合计算丢包率
    plr = base_plr + plr_distance + plr_wave + plr_ber + plr_oscillation + random_factor

    # 暴风影响 - 使用柔和的增强函数
    if env.storm_active:
        # 基于暴风持续时间的平滑过渡
        storm_progress = 1.0 - (env.storm_timer / env.storm_duration)
        # 使用平滑的钟形曲线
        storm_factor = 1.0 + 0.5 * np.sin(np.pi * storm_progress)
        # 乘以一个不太大的系数
        plr = plr * storm_factor + 0.1 * storm_factor  # 添加一个增量而不仅是乘法

    # 确保丢包率在合理范围内，防止过大
    plr = np.clip(plr, 0.001, 0.5)  # 将丢包率上限限制为50%，更符合实际情况

    return bandwidth, ber, plr


# 修改可视化函数，将暴风说明文本移至图表下方
def visualize_separate_charts(time, distances, wave_heights, bandwidths, bers, plrs, storm_periods, storm_start_times, storm_end_times):
    # 辅助函数：添加暴风区域并在下方注释
    def add_storm_regions(ax, ylims):
        # 标记暴风区域
        for i in range(len(storm_start_times)):
            start_time = storm_start_times[i]
            end_time = storm_end_times[i]
            # 添加红色阴影区域
            ax.axvspan(start_time, end_time, alpha=0.2, color='red')

            # 在图表下方添加暴风说明
            y_pos = ylims[0] - 0.15 * (ylims[1] - ylims[0])  # 在y轴下方15%位置
            ax.annotate('暴风', xy=((start_time + end_time)/2, y_pos), xycoords='data',
                        xytext=(0, -20), textcoords='offset points',
                        ha='center', va='top', color='red', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='red', alpha=0.7))

    # 第一张图：信道距离和风浪强度
    plt.figure(figsize=(12, 6))

    # 主纵轴 - 距离
    ax1 = plt.gca()
    ax1.plot(time, distances, 'b-', linewidth=2, label='节点间距离')
    ax1.set_xlabel('时间 (分钟)')
    ax1.set_ylabel('距离 (m)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 设置主y轴范围留出下方空间用于注释
    y_min = min(distances) * 0.9
    y_max = max(distances) * 1.1
    ax1.set_ylim(y_min, y_max)

    # 次纵轴 - 波浪高度
    ax2 = ax1.twinx()
    ax2.plot(time, wave_heights, 'r-', linewidth=2, label='波浪高度')
    ax2.set_ylabel('波浪高度 (m)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 标记暴风区域并在下方注释
    add_storm_regions(ax1, (y_min, y_max))

    # 在图例中添加暴风说明
    red_patch = mpatches.Patch(color='red', alpha=0.2, label='暴风区域')

    # 确保双轴的图例正确显示
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + [red_patch], labels1 + labels2 + ['暴风区域'],
               loc='upper right', framealpha=0.9)

    plt.title('节点间距离与波浪强度随时间变化')
    plt.grid(True, alpha=0.3)

    # 设置x轴范围留出空间
    plt.xlim(-0.5, max(time) + 0.5)

    plt.tight_layout()
    plt.savefig('distance_and_wave.png', dpi=300)

    # 第二张图：网络信道带宽
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(time, bandwidths / 1e6, 'g-', linewidth=2)

    # 设置y轴范围留出下方空间用于注释
    y_min = min(bandwidths / 1e6) * 0.9
    y_max = max(bandwidths / 1e6) * 1.1
    ax.set_ylim(y_min, y_max)

    # 标记暴风区域并在下方注释
    add_storm_regions(ax, (y_min, y_max))

    # 在图例中添加暴风说明
    red_patch = mpatches.Patch(color='red', alpha=0.2, label='暴风区域(带宽下降)')
    ax.legend(handles=[red_patch], loc='upper right', framealpha=0.9)

    plt.xlabel('时间 (分钟)')
    plt.ylabel('带宽 (MHz)')
    plt.title('网络信道带宽随时间变化')
    plt.grid(True, alpha=0.3)

    # 设置x轴范围留出空间
    plt.xlim(-0.5, max(time) + 0.5)

    plt.tight_layout()
    plt.savefig('bandwidth.png', dpi=300)

    # 第三张图：网络误码率
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.semilogy(time, bers, 'r-', linewidth=2)

    # 设置y轴范围留出下方空间用于注释
    y_min = min(bers) * 0.5
    y_max = max(bers) * 2.0
    ax.set_ylim(y_min, y_max)

    # 标记暴风区域并在下方注释
    add_storm_regions(ax, (y_min, y_max))

    # 在图例中添加暴风说明
    red_patch = mpatches.Patch(color='red', alpha=0.2, label='暴风区域(误码率上升)')
    ax.legend(handles=[red_patch], loc='upper right', framealpha=0.9)

    plt.xlabel('时间 (分钟)')
    plt.ylabel('误码率 (BER)')
    plt.title('网络误码率随时间变化')
    plt.grid(True, alpha=0.3)

    # 设置x轴范围留出空间
    plt.xlim(-0.5, max(time) + 0.5)

    plt.tight_layout()
    plt.savefig('ber.png', dpi=300)

    # 第四张图：网络丢包率
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(time, plrs, 'm-', linewidth=2)

    # 设置y轴范围留出下方空间用于注释
    y_min = 0
    y_max = max(plrs) * 1.2
    ax.set_ylim(y_min, y_max)

    # 标记暴风区域并在下方注释
    add_storm_regions(ax, (y_min, y_max))

    # 在图例中添加暴风说明
    red_patch = mpatches.Patch(color='red', alpha=0.2, label='暴风区域(丢包率上升)')
    ax.legend(handles=[red_patch], loc='upper right', framealpha=0.9)

    plt.xlabel('时间 (分钟)')
    plt.ylabel('丢包率 (PLR)')
    plt.title('网络丢包率随时间变化')
    plt.grid(True, alpha=0.3)

    # 设置x轴范围留出空间
    plt.xlim(-0.5, max(time) + 0.5)

    plt.tight_layout()
    plt.savefig('plr.png', dpi=300)

    # 显示所有图表
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
    wave_heights = np.zeros(time_steps)
    storm_periods = np.zeros(time_steps)  # 记录暴风状态

    # 主模拟循环
    for i in range(time_steps):
        t = i * dt

        # 更新天气状态
        env.update_weather(t)
        storm_periods[i] = 1 if env.storm_active else 0

        # 获取当前波浪高度
        _, current_wave_height = env.get_wave_effect(t, node1.position)
        wave_heights[i] = current_wave_height

        # 更新节点位置
        pos1, vel1 = node1.update(env, t, dt)
        pos2, vel2 = node2.update(env, t, dt)

        positions1[i] = pos1
        positions2[i] = pos2

        # 计算信道状态
        bandwidth, ber, plr = calculate_channel_state(node1, node2, env, t, current_wave_height)

        bandwidths[i] = bandwidth
        bers[i] = ber
        plrs[i] = plr
        distances[i] = np.linalg.norm(pos1 - pos2)

    # 可视化结果，生成四张独立图表
    visualize_separate_charts(time_minutes, distances, wave_heights, bandwidths, bers,
                             plrs, storm_periods, env.storm_start_times, env.storm_end_times)


# 程序入口
if __name__ == "__main__":
    print("开始海上浮标通信信道模拟(30分钟)...")
    run_simulation()
    print("模拟完成，结果已显示并保存。")
