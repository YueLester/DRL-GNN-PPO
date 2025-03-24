import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import math
from scipy.special import erfc


# Longley-Rice模型参数
def calculate_path_loss(distance, frequency, tx_height, rx_height, environment='suburban'):
    """
    使用简化的Longley-Rice模型计算路径损耗

    参数:
    distance: 距离(米)
    frequency: 频率(MHz)
    tx_height: 发射天线高度(米)
    rx_height: 接收天线高度(米)
    environment: 环境类型('urban', 'suburban', 'rural')

    返回:
    path_loss: 路径损耗(dB)
    """
    # 频率对数
    log_f = math.log10(frequency)

    # 距离对数
    log_d = math.log10(distance / 1000)  # 转换为千米

    # 环境因子
    if environment == 'urban':
        a = 69.55 + 26.16 * log_f - 13.82 * math.log10(tx_height)
        b = 44.9 - 6.55 * math.log10(tx_height)
        correction = 3  # 城市环境修正
    elif environment == 'suburban':
        a = 69.55 + 26.16 * log_f - 13.82 * math.log10(tx_height)
        b = 44.9 - 6.55 * math.log10(tx_height)
        correction = 0  # 郊区环境修正
    else:  # rural
        a = 69.55 + 26.16 * log_f - 13.82 * math.log10(tx_height)
        b = 44.9 - 6.55 * math.log10(tx_height)
        correction = -4.78 * (log_f ** 2) - 18.33 * log_f + 40.94  # 乡村环境修正

    # 计算路径损耗
    path_loss = a + b * log_d - 2 * (math.log10(rx_height) - 0.2) + correction

    return path_loss


# 计算信噪比
def calculate_snr(tx_power, path_loss, noise_figure, bandwidth, implementation_loss=0):
    """
    计算信噪比

    参数:
    tx_power: 发射功率(dBm)
    path_loss: 路径损耗(dB)
    noise_figure: 噪声系数(dB)
    bandwidth: 带宽(Hz)
    implementation_loss: 实现损耗(dB)

    返回:
    snr: 信噪比(dB)
    """
    # 计算热噪声功率 P_noise = k * T * B
    k = 1.38e-23  # 玻尔兹曼常数
    T = 290  # 温度(K)
    noise_power_dBm = 10 * math.log10(k * T * bandwidth * 1000) + noise_figure

    # 计算接收功率
    rx_power_dBm = tx_power - path_loss - implementation_loss

    # 计算信噪比
    snr = rx_power_dBm - noise_power_dBm

    return snr


# 计算误码率(BER)
def calculate_ber(snr_linear, modulation='QPSK'):
    """
    计算误码率

    参数:
    snr_linear: 线性信噪比(非dB)
    modulation: 调制方式

    返回:
    ber: 误码率
    """
    if modulation == 'BPSK':
        ber = 0.5 * erfc(np.sqrt(snr_linear))
    elif modulation == 'QPSK':
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))
    elif modulation == '16QAM':
        ber = 0.375 * erfc(np.sqrt(0.1 * snr_linear))
    elif modulation == '64QAM':
        ber = 0.1666 * erfc(np.sqrt(0.033 * snr_linear))
    else:
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))  # 默认QPSK

    return ber


# 计算丢包率(PER)
def calculate_per(ber, packet_size):
    """
    计算丢包率

    参数:
    ber: 误码率
    packet_size: 数据包大小(比特)

    返回:
    per: 丢包率
    """
    # 简化的PER计算模型: PER = 1 - (1-BER)^packet_size
    per = 1 - (1 - ber) ** packet_size

    # 限制PER最大值为1
    if per > 1:
        per = 1

    return per


# 主函数
def simulate_wireless_performance():
    # 参数设置
    tx_power = 30  # 发射功率(dBm)
    frequency = 2400  # 频率(MHz)
    tx_height = 30  # 发射天线高度(m)
    rx_height = 1.5  # 接收天线高度(m)
    noise_figure = 6  # 噪声系数(dB)
    implementation_loss = 3  # 实现损耗(dB)
    packet_size = 1024 * 8  # 数据包大小(比特)，1KB

    # 创建距离数组(100m到3000m)
    distances = np.linspace(100, 3000, 100)

    # 不同带宽和调制方式的组合
    bandwidths = [5e6, 10e6, 20e6]  # 5MHz, 10MHz, 20MHz
    modulations = ['BPSK', 'QPSK', '16QAM', '64QAM']

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 为每个带宽绘制曲线
    for bandwidth in bandwidths:
        # 选择合适的调制方式（根据带宽调整）
        if bandwidth == 5e6:
            modulation = 'QPSK'
        elif bandwidth == 10e6:
            modulation = '16QAM'
        else:
            modulation = '64QAM'

        # 计算每个距离点的性能指标
        ber_values = []
        per_values = []

        for distance in distances:
            # 计算路径损耗
            path_loss = calculate_path_loss(distance, frequency, tx_height, rx_height)

            # 计算信噪比
            snr_db = calculate_snr(tx_power, path_loss, noise_figure, bandwidth, implementation_loss)
            snr_linear = 10 ** (snr_db / 10)

            # 计算误码率
            ber = calculate_ber(snr_linear, modulation)
            ber_values.append(ber)

            # 计算丢包率
            per = calculate_per(ber, packet_size)
            per_values.append(per)

        # 绘制误码率曲线
        ax1.semilogy(distances, ber_values, label=f'{int(bandwidth / 1e6)}MHz带宽 ({modulation})')

        # 绘制丢包率曲线
        ax2.plot(distances, per_values, label=f'{int(bandwidth / 1e6)}MHz带宽 ({modulation})')

    # 设置误码率图的属性
    ax1.set_title('基于Longley-Rice模型的距离vs误码率关系')
    ax1.set_xlabel('距离 (米)')
    ax1.set_ylabel('误码率 (BER)')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    ax1.set_ylim(1e-8, 1)

    # 设置丢包率图的属性
    ax2.set_title('基于Longley-Rice模型的距离vs丢包率关系')
    ax2.set_xlabel('距离 (米)')
    ax2.set_ylabel('丢包率 (PER)')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


# 运行模拟
if __name__ == "__main__":
    simulate_wireless_performance()

