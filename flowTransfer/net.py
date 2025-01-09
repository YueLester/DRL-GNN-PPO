import math
import numpy as np


class WirelessMetrics:
    def __init__(self, frequency_ghz=2.4, tx_power_dbm=20, antenna_gain_db=2):
        """
        初始化无线通信性能计算类

        参数:
        frequency_ghz: 工作频率(GHz)
        tx_power_dbm: 发射功率(dBm)
        antenna_gain_db: 天线增益(dB)
        """
        self.frequency = frequency_ghz * 1e9  # 转换为Hz
        self.tx_power = tx_power_dbm
        self.antenna_gain = antenna_gain_db
        self.c = 3e8  # 光速(m/s)

    def calculate_path_loss(self, distance):
        """计算路径损耗(dB)"""
        # 使用自由空间路径损耗模型
        wavelength = self.c / self.frequency
        path_loss_db = 20 * math.log10(distance) + 20 * math.log10(self.frequency) - 147.55
        return path_loss_db

    def calculate_power_consumption(self, distance):
        """
        计算传输1字节数据所需的能量(μJ)
        考虑发射功率、接收电路功耗、传输时间等因素
        """
        # 基本参数设置
        byte_length = 8  # 1字节 = 8比特
        data_rate = 250 * 1000  # 假设数据传输率为250kbps

        # 电路功耗参数 (mW)
        circuit_power_tx = 15.0  # 发射电路功耗
        circuit_power_rx = 12.0  # 接收电路功耗

        # 计算发射功率
        path_loss = self.calculate_path_loss(distance)
        total_loss = path_loss - 2 * self.antenna_gain
        tx_power_mw = 10 ** ((self.tx_power + total_loss) / 10)

        # 计算传输时间 (s)
        transmission_time = byte_length / data_rate

        # 计算总能耗 (μJ)
        # 1. 发射端能耗：(发射功率 + 发射电路功耗) * 时间
        tx_energy = (tx_power_mw + circuit_power_tx) * transmission_time * 1000

        # 2. 接收端能耗：接收电路功耗 * 时间
        rx_energy = circuit_power_rx * transmission_time * 1000

        # 总能耗
        total_energy = tx_energy + rx_energy

        return total_energy

    def calculate_delay(self, distance, processing_delay_ms=250):
        """
        计算总延迟(ms)
        包括传播延迟、处理延迟和排队延迟
        目标控制在300ms左右
        """
        # 传播延迟
        propagation_delay_ms = (distance / self.c) * 1000

        # 排队延迟（假设与距离和网络负载相关）
        queue_delay_ms = 20 + (distance / 300) * 10

        # 总延迟 = 传播延迟 + 处理延迟 + 排队延迟
        total_delay_ms = propagation_delay_ms + processing_delay_ms + queue_delay_ms

        # 如果总延迟超过300ms，尝试调整处理延迟和排队延迟
        if total_delay_ms > 300:
            # 动态调整处理延迟和排队延迟
            scale_factor = 300 / total_delay_ms
            processing_delay_ms *= scale_factor
            queue_delay_ms *= scale_factor
            total_delay_ms = propagation_delay_ms + processing_delay_ms + queue_delay_ms

        return total_delay_ms

    def calculate_bandwidth(self, distance):
        """
        计算可用带宽(Mbps)
        基于香农定理，考虑信噪比
        """
        # 假设噪声功率为-90dBm
        noise_power_dbm = -90
        # 计算接收信号强度
        rx_power_dbm = self.tx_power - self.calculate_path_loss(distance) + 2 * self.antenna_gain
        # 计算信噪比
        snr = 10 ** ((rx_power_dbm - noise_power_dbm) / 10)
        # 使用香农定理计算理论带宽（假设带宽为20MHz）
        bandwidth_hz = 20e6
        capacity_bps = bandwidth_hz * math.log2(1 + snr)
        return capacity_bps / 1e6  # 转换为Mbps

    def calculate_packet_loss(self, distance):
        """
        计算丢包率(%)
        针对100-300m距离范围，控制丢包率在10%-20%之间
        使用距离的线性插值计算丢包率
        """
        # 定义距离范围和对应的丢包率范围
        min_distance = 100  # 最小距离
        max_distance = 300  # 最大距离
        min_loss = 10  # 最小丢包率
        max_loss = 20  # 最大丢包率

        # 对距离进行限制
        clamped_distance = max(min_distance, min(max_distance, distance))

        # 线性插值计算丢包率
        packet_loss = min_loss + (max_loss - min_loss) * (clamped_distance - min_distance) / (
                    max_distance - min_distance)

        # 如果距离超出范围，增加惩罚项
        if distance < min_distance:
            packet_loss = min_loss * (min_distance / max(distance, 1))
        elif distance > max_distance:
            packet_loss = max_loss * (distance / max_distance)

        return packet_loss


# 使用示例
def main():
    # 创建实例
    wireless = WirelessMetrics(frequency_ghz=2.4, tx_power_dbm=20, antenna_gain_db=2)

    # 测试目标距离范围内的性能指标
    distances = [100, 150, 200, 250, 300]
    print("距离(m) | 能耗(μJ) | 延迟(ms) | 带宽(Mbps) | 丢包率(%)")
    print("-" * 60)

    for distance in distances:
        power = wireless.calculate_power_consumption(distance)
        delay = wireless.calculate_delay(distance)
        bandwidth = wireless.calculate_bandwidth(distance)
        packet_loss = wireless.calculate_packet_loss(distance)

        print(f"{distance:7.1f} | {power:8.2f} | {delay:8.3f} | {bandwidth:9.2f} | {packet_loss:9.2f}")


if __name__ == "__main__":
    main()