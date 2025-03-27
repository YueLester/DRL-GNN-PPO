import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from matplotlib.dates import DateFormatter
import datetime as dt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 基本参数设置
freq = 2e9                # 通信频率 (Hz)
wavelength = 3e8 / freq   # 波长 (m)
distance = 10000          # 通信距离 (m)
ht0 = 15                  # 初始发射天线高度 (m)
hr0 = 10                  # 初始接收天线高度 (m)
simulation_hours = 24     # 仿真时长 (小时)
time_step = 10            # 时间步长 (分钟)

# 创建时间轴
time_points = int(simulation_hours * 60 / time_step) + 1
t_hours = np.linspace(0, simulation_hours, time_points)
t_axis = [dt.datetime(2023, 7, 1, 0, 0) + dt.timedelta(hours=h) for h in t_hours]

# 环境参数随时间变化
def wind_speed(t):
    """风速变化模型 (m/s)"""
    return 5 + 8 * np.sin(2 * np.pi * t / 12)

def rainfall_rate(t):
    """降雨率变化模型 (mm/h)"""
    return np.maximum(0, 15 * np.sin(2 * np.pi * t / 8 - np.pi/4))

def wave_height(v_w):
    """有效波高计算 (m)"""
    g = 9.8  # 重力加速度
    return 0.21 * v_w**2 / g

# 计算时变环境参数
v_w = wind_speed(t_hours)
rain = rainfall_rate(t_hours)
h_s = wave_height(v_w)

# 信道模型相关参数和函数
def reflection_coefficient(theta_i):
    """海面反射系数"""
    epsilon_r = complex(70, 60)  # 海水相对介电常数，简化处理
    return (np.sin(theta_i) - np.sqrt(epsilon_r - np.cos(theta_i)**2)) / \
           (np.sin(theta_i) + np.sqrt(epsilon_r - np.cos(theta_i)**2))

def tide_factor(t):
    """潮汐影响因子"""
    beta = 0.2
    f_t = 1/12.42  # 主要半日潮周期 (小时^-1)
    phi_t = 0
    return 1 + beta * np.sin(2 * np.pi * f_t * t + phi_t)

def wave_factor(v_w, theta_i):
    """风浪影响因子"""
    sigma_h = 0.0051 * v_w**2  # 海面高度标准差 (m)
    return np.exp(-8 * np.pi**2 * sigma_h**2 / wavelength**2 * np.sin(theta_i)**2)

def phase_difference(ht, hr, d, wavelength):
    """直射路径与反射路径的相位差"""
    return 4 * np.pi * ht * hr / (wavelength * d)

def rain_attenuation(R, freq_ghz):
    """降雨衰减计算 (dB/km)"""
    # 简化模型，适用于6GHz左右的频率
    k = 0.0051
    alpha = 1.42
    return k * R**alpha

# 计算时变信道特性
theta_i = np.arctan((ht0 + hr0) / distance)  # 简化处理，假设入射角恒定
gamma = reflection_coefficient(theta_i)
T_f = tide_factor(t_hours)
W_f = wave_factor(v_w, theta_i)
delta_phi = phase_difference(ht0, hr0, distance, wavelength)
rain_atten = np.array([rain_attenuation(r, freq/1e9) for r in rain])

# 自由空间路径损耗 (dB)
L_FS = 20 * np.log10(4 * np.pi * distance / wavelength)

# 总路径损耗 (dB)
L_total = 20 * np.log10(np.abs(1 + np.abs(gamma) * T_f * W_f * np.exp(1j * delta_phi))) + rain_atten * distance/1000

# 发射功率和天线增益
Pt_dBm = 30  # 发射功率 (dBm)
Gt_dB = 10   # 发射天线增益 (dB)
Gr_dB = 10   # 接收天线增益 (dB)

# 接收信噪比计算 (dB)
noise_figure_dB = 6  # 接收机噪声系数 (dB)
bandwidth_kHz = 1000  # 系统带宽 (kHz)
thermal_noise_dBm = -174 + 10*np.log10(bandwidth_kHz * 1000)  # 热噪声功率 (dBm)
SNR_dB = Pt_dBm + Gt_dB + Gr_dB - L_total - noise_figure_dB - thermal_noise_dBm

# 计算Rice因子
K0 = 10  # 理想条件下的Rice因子
lambda_R = 0.1  # 降雨影响系数
K = K0 * W_f * np.exp(-lambda_R * rain)

# 计算误码率 (BER)
SNR_linear = 10**(SNR_dB/10)
def Q_function(x):
    return 0.5 * special.erfc(x/np.sqrt(2))

BER = np.array([Q_function(np.sqrt(2*snr/(k+1))) for snr, k in zip(SNR_linear, K)])

# 计算丢包率 (PER)
packet_length = 1000  # 数据包长度 (比特)
PER = 1 - (1 - BER)**packet_length

# 计算相干带宽
tau_rms0 = 1e-6  # 理想条件下的多径时延扩展 (秒)
eta = 0.3        # 波高影响系数
coherence_bandwidth = (1/(5*tau_rms0)) * np.exp(-eta * h_s) / 1e6  # MHz

# 绘制图像
plt.figure(figsize=(15, 20))

# 创建子图1：环境参数
plt.subplot(4, 1, 1)
ax1 = plt.gca()
line1, = ax1.plot(t_axis, v_w, 'b-', label='风速 (m/s)')
ax1.set_ylabel('风速 (m/s)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
line2, = ax2.plot(t_axis, rain, 'r-', label='降雨率 (mm/h)')
ax2.set_ylabel('降雨率 (mm/h)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
line3, = ax3.plot(t_axis, h_s, 'g-', label='有效波高 (m)')
ax3.set_ylabel('有效波高 (m)', color='g')
ax3.tick_params(axis='y', labelcolor='g')

lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
plt.title('海上环境参数随时间变化')
plt.legend(lines, labels, loc='upper right')

# 创建子图2：相干带宽
plt.subplot(4, 1, 2)
plt.plot(t_axis, coherence_bandwidth, 'b-')
plt.title('信道相干带宽随时间变化')
plt.ylabel('相干带宽 (MHz)')
plt.grid(True)

# 创建子图3：误码率
plt.subplot(4, 1, 3)
plt.semilogy(t_axis, BER, 'r-')
plt.title('误码率(BER)随时间变化')
plt.ylabel('误码率')
plt.grid(True)
plt.ylim([1e-6, 1e-2])

# 创建子图4：丢包率
plt.subplot(4, 1, 4)
plt.plot(t_axis, PER*100, 'g-')
plt.title('丢包率(PER)随时间变化')
plt.ylabel('丢包率 (%)')
plt.xlabel('时间')
plt.grid(True)
plt.ylim([0, 100])

# 设置x轴时间格式
for ax in plt.gcf().axes:
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('maritime_channel_quality.png', dpi=300)
plt.show()

# 输出一些关键统计数据
print(f"平均相干带宽: {np.mean(coherence_bandwidth):.2f} MHz")
print(f"最小相干带宽: {np.min(coherence_bandwidth):.2f} MHz")
print(f"平均误码率: {np.mean(BER):.2e}")
print(f"最大误码率: {np.max(BER):.2e}")
print(f"平均丢包率: {np.mean(PER)*100:.2f}%")
print(f"最大丢包率: {np.max(PER)*100:.2f}%")
