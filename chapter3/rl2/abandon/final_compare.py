
import numpy as np
import matplotlib.pyplot as plt
from path import Path
from trans1 import NetworkTransmissionSimulator
from typing import List, Dict, Tuple
import math
from matplotlib.gridspec import GridSpec
import time

    # 实现AMRBEC算法 - 基于不准的预测，没有探测开销


class AMRBECSimulator(NetworkTransmissionSimulator):
    """
    AMRBEC (Adaptive Multi-path Redundancy Based on Estimation with Channel prediction)

    基于不准的预测，没有探测开销，但是预测的网络信道状态与实际有出入，
    根据有偏差的方法做出网络决策。预测误差随网络波动程度而变化。
    """

    def __init__(self, path1: Path, path2: Path, **kwargs):
        super().__init__(path1, path2, **kwargs)
        # 初始化基础预测误差参数
        self.base_bw1_prediction_error = 0.15  # 带宽基础预测误差比例
        self.base_bw2_prediction_error = 0.20  # 带宽基础预测误差比例（路径2更大）
        self.base_pl1_prediction_error = 0.10  # 丢包率基础预测误差比例
        self.base_pl2_prediction_error = 0.15  # 丢包率基础预测误差比例（路径2更大）

        # 当前预测误差（会根据网络波动程度动态调整）
        self.bw1_prediction_error = self.base_bw1_prediction_error
        self.bw2_prediction_error = self.base_bw2_prediction_error
        self.pl1_prediction_error = self.base_pl1_prediction_error
        self.pl2_prediction_error = self.base_pl2_prediction_error

        # 趋势预测相关参数
        self.bw1_trend = 0
        self.bw2_trend = 0
        self.pl1_trend = 0
        self.pl2_trend = 0

        # 时间和状态跟踪
        self.last_update_time = 0
        self.network_variation_level1 = 1  # 网络变化等级（1-5）
        self.network_variation_level2 = 1  # 网络变化等级（1-5）

        # 上次估计的状态
        self.prev_estimated_bw1 = None
        self.prev_estimated_bw2 = None
        self.prev_estimated_pl1 = None
        self.prev_estimated_pl2 = None

    def _decide_probe(self, current_time: float, path_id: int) -> bool:
        """
        AMRBEC不进行探测，仅依赖于初始估计和预测模型
        只在开始时进行一次探测
        """
        return current_time == 0

    def _calculate_network_variation(self, current_time: float):
        """
        计算网络变化等级（1-5），用于调整预测误差
        网络变化越大，预测误差越大
        """
        # 获取当前的真实网络状态（仅用于模拟评估，实际系统中无法获取）
        current_state1 = self._get_path_state(current_time, self.path1)
        current_state2 = self._get_path_state(current_time, self.path2)

        # 计算与上次估计的差异
        if self.prev_estimated_bw1 is not None:
            # 路径1的变化量
            bw1_change = abs(current_state1['bandwidth'] - self.prev_estimated_bw1) / max(1.0, self.prev_estimated_bw1)
            pl1_change = abs(current_state1['packet_loss'] - self.prev_estimated_pl1) / max(0.01,
                                                                                            self.prev_estimated_pl1)

            # 路径2的变化量
            bw2_change = abs(current_state2['bandwidth'] - self.prev_estimated_bw2) / max(1.0, self.prev_estimated_bw2)
            pl2_change = abs(current_state2['packet_loss'] - self.prev_estimated_pl2) / max(0.01,
                                                                                            self.prev_estimated_pl2)

            # 计算综合变化量
            variation1 = max(bw1_change, pl1_change * 10)  # 丢包率变化权重更高
            variation2 = max(bw2_change, pl2_change * 10)

            # 更新网络变化等级（1-5）
            self.network_variation_level1 = min(5, max(1, int(variation1 * 10) + 1))
            self.network_variation_level2 = min(5, max(1, int(variation2 * 10) + 1))

        # 记录当前估计值，供下次计算变化量
        self.prev_estimated_bw1 = current_state1['bandwidth']
        self.prev_estimated_bw2 = current_state2['bandwidth']
        self.prev_estimated_pl1 = current_state1['packet_loss']
        self.prev_estimated_pl2 = current_state2['packet_loss']

    def _update_path_estimates(self, current_time: float, probe_path1: bool, probe_path2: bool):
        """
        更新路径估计，主要基于预测模型而非实际探测
        预测误差随网络变化程度动态调整
        """
        # 首先调用父类方法处理初始化和实际探测
        super()._update_path_estimates(current_time, probe_path1, probe_path2)

        # 计算网络变化程度
        self._calculate_network_variation(current_time)

        # 根据网络变化调整预测误差
        self.bw1_prediction_error = self.base_bw1_prediction_error * (self.network_variation_level1 / 2.5)
        self.bw2_prediction_error = self.base_bw2_prediction_error * (self.network_variation_level2 / 2.5)
        self.pl1_prediction_error = self.base_pl1_prediction_error * (self.network_variation_level1 / 2.5)
        self.pl2_prediction_error = self.base_pl2_prediction_error * (self.network_variation_level2 / 2.5)

        # 如果不是探测，则使用预测模型更新估计
        if current_time > 0 and not probe_path1 and not probe_path2:
            time_delta = current_time - self.last_update_time

            # 获取实际的网络状态（仅用于模拟，实际系统中无法获得）
            actual_state1 = self._get_path_state(current_time, self.path1)
            actual_state2 = self._get_path_state(current_time, self.path2)

            # 预测带宽变化 - 误差随网络变化程度增加
            if self.current_bandwidth1 is not None:
                # 添加预测误差 - 在实际值附近添加一个有偏差的估计
                ideal_prediction = actual_state1['bandwidth']
                error_factor = np.random.normal(0, self.bw1_prediction_error)
                self.current_bandwidth1 = ideal_prediction * (1 + error_factor)
                self.current_bandwidth1 = max(0.1, self.current_bandwidth1)  # 确保为正

            if self.current_bandwidth2 is not None:
                ideal_prediction = actual_state2['bandwidth']
                error_factor = np.random.normal(0, self.bw2_prediction_error)
                self.current_bandwidth2 = ideal_prediction * (1 + error_factor)
                self.current_bandwidth2 = max(0.1, self.current_bandwidth2)  # 确保为正

            # 预测丢包率变化 - 误差随网络变化程度增加
            if self.current_packet_loss1 is not None:
                ideal_prediction = actual_state1['packet_loss']
                error_factor = np.random.normal(0, self.pl1_prediction_error)
                self.current_packet_loss1 = ideal_prediction * (1 + error_factor)
                self.current_packet_loss1 = np.clip(self.current_packet_loss1, 0.001, 0.999)

            if self.current_packet_loss2 is not None:
                ideal_prediction = actual_state2['packet_loss']
                error_factor = np.random.normal(0, self.pl2_prediction_error)
                self.current_packet_loss2 = ideal_prediction * (1 + error_factor)
                self.current_packet_loss2 = np.clip(self.current_packet_loss2, 0.001, 0.999)

        self.last_update_time = current_time

    def _decide_transmission(self, current_time: float) -> Dict:
        """
        重写传输决策方法，根据预测误差动态调整冗余率
        网络变化大时增加冗余以应对不确定性
        """
        # 获取当前路径状态（用于评估）
        actual_state1 = self._get_path_state(current_time, self.path1)
        actual_state2 = self._get_path_state(current_time, self.path2)

        # 设置数据包数量(N)
        N = self.max_packets

        # 基于预测的丢包率和网络变化程度决定冗余
        pl1 = self.current_packet_loss1
        pl2 = self.current_packet_loss2

        # 计算基础冗余比例
        base_redundancy_ratio = (pl1 + pl2) / 2 * 1.5  # 基础冗余比例，考虑平均丢包率加上50%的安全边际

        # 根据网络变化程度增加额外冗余
        variation_factor = (self.network_variation_level1 + self.network_variation_level2) / 2
        extra_redundancy_ratio = 0.05 * (variation_factor - 1)  # 网络变化每增加一级，增加5%冗余

        # 总冗余比例，限制在合理范围内
        total_redundancy_ratio = np.clip(base_redundancy_ratio + extra_redundancy_ratio, 0.1, 0.5)

        # 计算冗余包数量
        R = int(N * total_redundancy_ratio / (1 - total_redundancy_ratio))
        R = min(R, self.max_redundancy)  # 确保不超过最大冗余

        # 计算数据包分配
        w1, w2 = self._calculate_packet_distribution(
            N, R, self.current_bandwidth1, self.current_bandwidth2
        )

        # 计算预期成功率（使用实际路径状态）
        success_rate = self._calculate_success_rate(
            N, R, w1, w2,
            actual_state1['packet_loss'],
            actual_state2['packet_loss']
        )

        # 计算冗余率
        redundancy_ratio = R / (N + R) if (N + R) > 0 else 0

        return {
            'N': N,
            'R': R,
            'w1': w1,
            'w2': w2,
            'bandwidth1': actual_state1['bandwidth'],
            'bandwidth2': actual_state2['bandwidth'],
            'packet_loss1': actual_state1['packet_loss'],
            'packet_loss2': actual_state2['packet_loss'],
            'success_rate': success_rate,
            'redundancy_ratio': redundancy_ratio
        }


# 实现AAEEMR算法 - 不采用网络编码，使用多路径进行传输，针对丢包进行重传
class AAEEMRSimulator(NetworkTransmissionSimulator):
    """
    AAEEMR (Adaptive Allocation with Energy-Efficient Multi-path Retransmission)

    不采用网络编码，使用多路径进行传输，针对网络中存在的丢包进行重传
    """

    def __init__(self, path1: Path, path2: Path, **kwargs):
        super().__init__(path1, path2, **kwargs)
        # 由于不使用网络编码，重定义重传策略
        self.retransmission_probability = 0  # 用于跟踪预期的重传概率
        self.required_packets = 0  # 解码所需的最少包数
        self.missing_packets = 0  # 重传所需的包数量
        self.last_probe_interval = 120  # 探测间隔更长（120秒）

    def _decide_probe(self, current_time: float, path_id: int) -> bool:
        """
        AAEEMR使用较低频率的探测，因为它不需要精确的丢包率估计，
        只需要大致了解链路状况用于流量分配
        """
        # 初始化探测
        if current_time == 0:
            return True

        # 获取上次探测时间
        if path_id == 1:
            last_probe_time = self.last_probe_time1
        else:
            last_probe_time = self.last_probe_time2

        # 只有当时间间隔足够长时才进行探测
        return (current_time - last_probe_time) >= self.last_probe_interval

    def _decide_transmission(self, current_time: float) -> Dict:
        """
        重写传输决策，不使用网络编码，而是根据两条路径状态分配数据包
        """
        # 获取当前路径状态（用于评估）
        actual_state1 = self._get_path_state(current_time, self.path1)
        actual_state2 = self._get_path_state(current_time, self.path2)

        # 设置数据包数量(N)为最大值
        N = self.max_packets
        self.required_packets = N  # 需要接收到全部N个包才能解码

        # 由于不使用网络编码，所以不设置冗余包
        R = 0

        # 根据带宽比例分配数据包
        total_bandwidth = self.current_bandwidth1 + self.current_bandwidth2
        if total_bandwidth > 0:
            w1_ratio = self.current_bandwidth1 / total_bandwidth
        else:
            w1_ratio = 0.5

        # 计算每条路径上的包分配
        w1 = int(round(N * w1_ratio))
        w2 = N - w1

        # 确保每条路径上至少有1个包（如果带宽非零）
        if w1 == 0 and self.current_bandwidth1 > 0:
            w1 = 1
            w2 = N - 1
        elif w2 == 0 and self.current_bandwidth2 > 0:
            w2 = 1
            w1 = N - 1

        # 计算预期接收包数
        expected_received_path1 = w1 * (1 - actual_state1['packet_loss'])
        expected_received_path2 = w2 * (1 - actual_state2['packet_loss'])
        expected_received_total = expected_received_path1 + expected_received_path2

        # 计算缺少的包数（需要重传的包数）
        self.missing_packets = max(0, N - int(expected_received_total))

        # 计算重传概率
        if N > 0:
            self.retransmission_probability = self.missing_packets / N
        else:
            self.retransmission_probability = 0

        # 计算实际成功率（使用实际丢包率）- 不考虑重传的情况
        success_rate = expected_received_total / N if N > 0 else 0

        # 计算冗余率 - 由于没有冗余包，所以为0
        redundancy_ratio = 0

        return {
            'N': N,
            'R': R,
            'w1': w1,
            'w2': w2,
            'bandwidth1': actual_state1['bandwidth'],
            'bandwidth2': actual_state2['bandwidth'],
            'packet_loss1': actual_state1['packet_loss'],
            'packet_loss2': actual_state2['packet_loss'],
            'success_rate': success_rate,
            'redundancy_ratio': redundancy_ratio
        }


# 实现RMPR算法 - 使用副本机制传输，使用固定冗余率
class RMPRSimulator(NetworkTransmissionSimulator):
    """
    RMPR (Redundant Multi-Path Replication)

    使用副本机制传输，使用50%的固定冗余率，无探测
    """

    def __init__(self, path1: Path, path2: Path, redundancy_ratio: float = 0.5, **kwargs):
        """
        初始化，设置固定的冗余率

        Args:
            path1: 第一条通信路径
            path2: 第二条通信路径
            redundancy_ratio: 固定冗余率 (0-1)
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(path1, path2, **kwargs)
        self.redundancy_ratio = redundancy_ratio

    def _decide_probe(self, current_time: float, path_id: int) -> bool:
        """
        RMPR不进行网络探测，仅在初始时刻进行一次探测以获取初始状态
        """
        return current_time == 0

    def _decide_transmission(self, current_time: float) -> Dict:
        """
        使用固定冗余率进行传输决策
        """
        # 获取实际路径状态（用于评估）
        actual_state1 = self._get_path_state(current_time, self.path1)
        actual_state2 = self._get_path_state(current_time, self.path2)

        # 设置数据包数量(N)
        N = self.max_packets

        # 根据固定冗余率计算冗余包数量(R)
        R = int(N * self.redundancy_ratio / (1 - self.redundancy_ratio))
        R = min(R, self.max_redundancy)  # 确保不超过最大冗余

        # 计算数据包分配
        w1, w2 = self._calculate_packet_distribution(
            N, R, self.current_bandwidth1, self.current_bandwidth2
        )

        # 计算预期成功率（使用实际路径状态）
        success_rate = self._calculate_success_rate(
            N, R, w1, w2,
            actual_state1['packet_loss'],
            actual_state2['packet_loss']
        )

        # 计算冗余率
        redundancy_ratio = R / (N + R) if (N + R) > 0 else 0

        return {
            'N': N,
            'R': R,
            'w1': w1,
            'w2': w2,
            'bandwidth1': actual_state1['bandwidth'],
            'bandwidth2': actual_state2['bandwidth'],
            'packet_loss1': actual_state1['packet_loss'],
            'packet_loss2': actual_state2['packet_loss'],
            'success_rate': success_rate,
            'redundancy_ratio': redundancy_ratio
        }


# 实现AFB-GPSR算法 - 根据网络状态使用固定网络探测率
class AFBGPSRSimulator(NetworkTransmissionSimulator):
    """
    AFB-GPSR (Adaptive Feedback with Geographical Position Statistic Routing)

    根据网络状态使用固定网络探测率
    """

    def __init__(self, path1: Path, path2: Path, probe_frequency: int = 5, **kwargs):
        """
        初始化，设置固定的探测频率

        Args:
            path1: 第一条通信路径
            path2: 第二条通信路径
            probe_frequency: 固定探测频率（每X个决策点进行一次探测）
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(path1, path2, **kwargs)
        self.probe_frequency = probe_frequency
        self.decision_count = 0

    def _decide_probe(self, current_time: float, path_id: int) -> bool:
        """
        使用固定频率进行探测决策
        """
        # 初始化探测
        if current_time == 0:
            return True

        # 仅当决策计数是探测频率的倍数时才进行探测
        if path_id == 1:  # 只在path_id=1时增加计数，避免重复
            self.decision_count += 1

        return self.decision_count % self.probe_frequency == 0


# 主函数：运行所有算法的对比分析
def run_full_algorithm_comparison():
    """
    运行完整的算法对比分析，包括所有实现的算法
    """
    print("开始海上网络传输算法对比分析...")

    # 创建具有不同特性的路径模拟海上通信网络
    # 路径1: 更稳定的卫星链路，带宽更高，丢包率更低
    # 路径2: 不稳定的RF链路，带宽较低，丢包率较高

    # 设置3小时仿真 (180分钟)，共36个时隙(每个5分钟)
    # 第1小时 (0-60分钟): 平稳
    # 第2小时 (60-120分钟): 略有变化
    # 第3小时 (120-180分钟): 变化激烈
    total_slots = 36

    # 路径1参数 (卫星链路)
    bw1_avg = []
    pl1_avg = []
    fluct1_level = []

    # 第1小时 (0-60分钟): 平稳状态 (波动等级1-2)
    for i in range(12):
        bw1_avg.append(12 + np.random.uniform(-1, 1))  # 稳定在约12 Mbps左右
        pl1_avg.append(0.03 + np.random.uniform(-0.01, 0.01))  # 稳定在约3%丢包率
        fluct1_level.append(np.random.randint(1, 3))  # 波动等级1-2

    # 第2小时 (60-120分钟): 略有变化 (波动等级2-3)
    for i in range(12):
        bw1_avg.append(11 + np.random.uniform(-2, 2))  # 11 Mbps左右，略有波动
        pl1_avg.append(0.04 + np.random.uniform(-0.02, 0.02))  # 约4%丢包率，略有波动
        fluct1_level.append(np.random.randint(2, 4))  # 波动等级2-3

    # 第3小时 (120-180分钟): 变化激烈 (波动等级3-5)
    for i in range(12):
        bw1_avg.append(10 + np.random.uniform(-3, 3))  # 10 Mbps左右，波动较大
        pl1_avg.append(0.05 + np.random.uniform(-0.03, 0.03))  # 约5%丢包率，波动较大
        fluct1_level.append(np.random.randint(3, 6))  # 波动等级3-5

    # 路径2参数 (RF链路)
    bw2_avg = []
    pl2_avg = []
    fluct2_level = []

    # 第1小时 (0-60分钟): 平稳状态 (波动等级1-3)
    for i in range(12):
        bw2_avg.append(8 + np.random.uniform(-1, 1))  # 稳定在约8 Mbps左右
        pl2_avg.append(0.08 + np.random.uniform(-0.01, 0.01))  # 稳定在约8%丢包率
        fluct2_level.append(np.random.randint(1, 4))  # 波动等级1-3

    # 第2小时 (60-120分钟): 略有变化 (波动等级2-4)
    for i in range(12):
        bw2_avg.append(7 + np.random.uniform(-2, 2))  # 7 Mbps左右，略有波动
        pl2_avg.append(0.10 + np.random.uniform(-0.02, 0.02))  # 约10%丢包率，略有波动
        fluct2_level.append(np.random.randint(2, 5))  # 波动等级2-4

    # 第3小时 (120-180分钟): 变化激烈 (波动等级3-5)
    for i in range(12):
        bw2_avg.append(6 + np.random.uniform(-3, 3))  # 6 Mbps左右，波动较大
        pl2_avg.append(0.12 + np.random.uniform(-0.04, 0.04))  # 约12%丢包率，波动较大
        fluct2_level.append(np.random.randint(3, 6))  # 波动等级3-5

    # 确保所有丢包率在合理范围内
    pl1_avg = [max(0.01, min(0.2, pl)) for pl in pl1_avg]
    pl2_avg = [max(0.02, min(0.3, pl)) for pl in pl2_avg]

    # 确保所有带宽在合理范围内
    bw1_avg = [max(1, bw) for bw in bw1_avg]
    bw2_avg = [max(1, bw) for bw in bw2_avg]

    # 创建两条路径，使用不同的波动模式
    path1 = Path(
        avg_bandwidth_series=bw1_avg,
        avg_packet_loss_series=pl1_avg,
        fluctuation_level_series=fluct1_level,
        time_slot_duration=5,  # 每个时隙5分钟
        points_per_slot=300,  # 每个时隙300个点 (每秒1个点)
        fluctuation_pattern='sine'  # 路径1使用正弦波模式
    )

    path2 = Path(
        avg_bandwidth_series=bw2_avg,
        avg_packet_loss_series=pl2_avg,
        fluctuation_level_series=fluct2_level,
        time_slot_duration=5,  # 每个时隙5分钟
        points_per_slot=300,  # 每个时隙300个点 (每秒1个点)
        fluctuation_pattern='mixed'  # 路径2使用混合模式
    )

    # 公共模拟器参数
    sim_params = {
        'decision_interval': 10,  # 每10秒做一次决策
        'simulation_duration': 180,  # 总共仿真180分钟
        'max_packets': 64,  # 最大数据包数量(N)
        'max_redundancy': 24,  # 最大冗余包数量(R)
        'probe_interval': 30  # 探测的最小间隔(秒)
    }

    # 创建对比对象
    comparison = AlgorithmComparison(path1, path2, simulation_duration=180)

    # 添加算法进行对比
    # 1. AMMPNC 算法 (原文算法)
    ammpnc_algorithm = NetworkTransmissionSimulator(
        path1=path1,
        path2=path2,
        **sim_params
    )
    comparison.add_algorithm("AMMPNC", ammpnc_algorithm)

    # 2. AMRBEC 算法
    amrbec_algorithm = AMRBECSimulator(
        path1=path1,
        path2=path2,
        **sim_params
    )
    comparison.add_algorithm("AMRBEC", amrbec_algorithm)

    # 3. AAEEMR 算法
    aaeemr_algorithm = AAEEMRSimulator(
        path1=path1,
        path2=path2,
        **sim_params
    )
    comparison.add_algorithm("AAEEMR", aaeemr_algorithm)

    # 4. RMPR 算法
    rmpr_algorithm = RMPRSimulator(
        path1=path1,
        path2=path2,
        redundancy_ratio=0.5,  # 50%的固定冗余率
        **sim_params
    )
    comparison.add_algorithm("RMPR", rmpr_algorithm)

    # 5. AFB-GPSR 算法
    afbgpsr_algorithm = AFBGPSRSimulator(
        path1=path1,
        path2=path2,
        probe_frequency=5,  # 每5个决策点进行一次探测
        **sim_params
    )
    comparison.add_algorithm("AFB-GPSR", afbgpsr_algorithm)

    # 运行对比
    print("\n运行算法对比...")
    comparison.run_comparison()

    # 打印结果摘要
    comparison.print_summary()

    # 绘制对比图
    print("\n生成对比图...")
    fig_comparison = comparison.plot_comparison(figsize=(14, 16))
    plt.savefig('算法对比图.png', dpi=300, bbox_inches='tight')

    # 为每个算法单独绘制结果图
    print("\n生成单独算法结果图...")
    for algorithm in comparison.algorithms.keys():
        fig_alg = comparison.plot_individual_algorithm(algorithm, figsize=(10, 14))
        plt.savefig(f'{algorithm}结果图.png', dpi=300, bbox_inches='tight')

    print("\n分析完成，图表已保存。")

    # 返回对比对象以供进一步分析
    return comparison


class AlgorithmComparison:
    """
    比较不同网络传输算法的性能。
    """

    def __init__(self, path1: Path, path2: Path, simulation_duration: int = 180):
        """
        初始化比较对象。

        Args:
            path1: 第一条通信路径
            path2: 第二条通信路径
            simulation_duration: 仿真时长（分钟）
        """
        self.path1 = path1
        self.path2 = path2
        self.simulation_duration = simulation_duration
        self.algorithms = {}
        self.results = {}

        # 能耗参数
        self.probe_energy = 0.5  # 每次探测的能耗(J)
        self.packet_size = 1500  # 每个数据包的大小(Bytes)
        self.retransmission_request_energy = 0.3  # 重传请求能耗(J)

    def add_algorithm(self, name: str, simulator):
        """
        添加一个算法到比较中。

        Args:
            name: 算法名称
            simulator: 算法模拟器实例
        """
        self.algorithms[name] = simulator

    def run_comparison(self):
        """
        运行所有算法并收集性能指标。
        """
        for name, simulator in self.algorithms.items():
            print(f"正在运行仿真: {name}")
            simulator.simulate()

            # 收集结果
            self.results[name] = self._calculate_metrics(simulator)

    def _calculate_metrics(self, simulator):
        """
        计算模拟器的性能指标。
        - 完成率：保持原始决策间隔
        - 其他指标：按5分钟间隔聚合

        Args:
            simulator: 已完成仿真的模拟器

        Returns:
            包含性能指标的字典
        """
        # 获取时间点数量
        num_points = len(simulator.transmission_history['times'])

        # 从传输历史中提取值
        times = simulator.transmission_history['times']
        completion_rate = simulator.transmission_history['success_rate']
        redundancy_ratio = simulator.transmission_history['redundancy_ratio']

        # 计算冗余数据量（不参与解码的数据）
        redundant_data_raw = []
        for i in range(num_points):
            N = simulator.transmission_history['N'][i]
            R = simulator.transmission_history['R'][i]
            success_rate = simulator.transmission_history['success_rate'][i]

            # 计算解码所需的预期包数
            expected_needed = N

            # 计算发送的总包数
            total_sent = N + R

            # 根据实际丢包率计算预期接收的包数
            pl1 = simulator.transmission_history['packet_loss1'][i]
            pl2 = simulator.transmission_history['packet_loss2'][i]
            w1 = simulator.transmission_history['w1'][i]
            w2 = simulator.transmission_history['w2'][i]

            # 从每条路径预期接收的包数
            expected_received_path1 = w1 * (1 - pl1)
            expected_received_path2 = w2 * (1 - pl2)
            expected_received_total = expected_received_path1 + expected_received_path2

            # 冗余数据是超过解码所需的部分
            # 这代表不参与解码的数据包
            redundant = max(0, expected_received_total - expected_needed)
            redundant_data_raw.append(redundant * self.packet_size)  # 转换为字节

        # 计算重传能耗
        retransmission_energy_raw = []
        for i in range(num_points):
            success_rate = simulator.transmission_history['success_rate'][i]
            # 如果传输失败需要重传的能耗
            energy = 0
            if success_rate < 0.999:  # 如果不是几乎必然成功
                failure_prob = 1 - success_rate
                energy = failure_prob * self.retransmission_request_energy
            retransmission_energy_raw.append(energy)

        # 计算控制能耗（来自探测）
        control_energy_raw = []
        probe_times = simulator.probe_history['times']
        for t in times:
            # 找出此时间点的所有探测
            energy = 0
            for i, probe_time in enumerate(probe_times):
                if probe_time == t:
                    if simulator.probe_history['path1'][i]:
                        energy += self.probe_energy
                    if simulator.probe_history['path2'][i]:
                        energy += self.probe_energy
            control_energy_raw.append(energy)

        # 按5分钟间隔分组指标（除了完成率）
        # 5分钟 = 300秒
        interval_size = 300  # 秒
        max_time = max(times)
        interval_count = int(np.ceil(max_time / interval_size)) + 1

        # 初始化聚合数据结构
        aggregated_times = [i * interval_size / 60.0 for i in range(interval_count)]  # 转换为分钟
        aggregated_redundant_data = [[] for _ in range(interval_count)]
        aggregated_retransmission_energy = [[] for _ in range(interval_count)]
        aggregated_control_energy = [[] for _ in range(interval_count)]

        # 将数据点分配到间隔中（除完成率外）
        for i in range(num_points):
            t = times[i]
            interval_idx = int(t / interval_size)

            aggregated_redundant_data[interval_idx].append(redundant_data_raw[i])
            aggregated_retransmission_energy[interval_idx].append(retransmission_energy_raw[i])
            aggregated_control_energy[interval_idx].append(control_energy_raw[i])

        # 计算每个间隔的平均值（除完成率外）
        avg_redundant_data = []
        avg_retransmission_energy = []
        avg_control_energy = []

        for interval_idx in range(interval_count):
            if aggregated_redundant_data[interval_idx]:
                avg_redundant_data.append(np.mean(aggregated_redundant_data[interval_idx]))
            else:
                avg_redundant_data.append(0)

            if aggregated_retransmission_energy[interval_idx]:
                avg_retransmission_energy.append(np.mean(aggregated_retransmission_energy[interval_idx]))
            else:
                avg_retransmission_energy.append(0)

            if aggregated_control_energy[interval_idx]:
                avg_control_energy.append(np.mean(aggregated_control_energy[interval_idx]))
            else:
                avg_control_energy.append(0)

        # 转换时间从秒到分钟（用于绘图）
        times_minutes = np.array(times) / 60.0

        # 返回指标 - 完成率保持原始分辨率，其他指标按5分钟聚合
        return {
            'original_times': times_minutes,  # 原始时间（分钟）
            'times': aggregated_times,  # 聚合时间（分钟）
            'completion_rate': completion_rate,  # 保持原始完成率
            'redundant_data': avg_redundant_data,  # 聚合冗余数据
            'retransmission_energy': avg_retransmission_energy,  # 聚合重传能耗
            'control_energy': avg_control_energy  # 聚合控制能耗
        }

    def plot_comparison(self, figsize=(14, 18)):
        """
        绘制算法之间的性能对比图：
        - 完成率：原始决策间隔数据
        - 其他指标：5分钟间隔数据

        Args:
            figsize: 图形大小，格式为(宽度, 高度)，单位为英寸
        """
        if not self.results:
            raise ValueError("没有可绘制的结果。请先运行对比。")

        # 创建带有子图的图形
        fig, axs = plt.subplots(4, 1, figsize=figsize)

        # 设置中文字体（如果可用）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于中文字符
            plt.rcParams['axes.unicode_minus'] = False  # 修复负号"-"显示问题
            plt.rcParams['font.family'] = 'SimHei'
        except:
            pass  # 如果字体不可用，继续使用默认字体

        # 算法的线型和颜色
        styles = ['-', '--', '-.', ':', '-']
        colors = ['b', 'r', 'g', 'm', 'c']

        # 1. 绘制完成率（不考虑重传）- 使用原始数据点
        ax = axs[0]
        for i, (name, result) in enumerate(self.results.items()):
            style = styles[i % len(styles)]
            color = colors[i % len(colors)]
            ax.plot(result['original_times'],
                    [rate * 100 for rate in result['completion_rate']],
                    linestyle=style, color=color, label=name)
        ax.set_ylabel('完成率 (%)')
        ax.set_title('无重传情况下的数据完成率')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. 绘制冗余数据量（5分钟间隔）
        ax = axs[1]
        for i, (name, result) in enumerate(self.results.items()):
            style = styles[i % len(styles)]
            color = colors[i % len(colors)]
            # 转换为KB以便更好地阅读
            redundant_kb = [r / 1024 for r in result['redundant_data']]
            ax.plot(result['times'], redundant_kb,
                    linestyle=style, color=color, label=name,
                    marker='o', markersize=5)
        ax.set_ylabel('冗余数据量 (KB)')
        ax.set_title('对解码无增益的冗余数据量 (5分钟平均)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 3. 绘制重传能耗（5分钟间隔）
        ax = axs[2]
        for i, (name, result) in enumerate(self.results.items()):
            style = styles[i % len(styles)]
            color = colors[i % len(colors)]
            ax.plot(result['times'], result['retransmission_energy'],
                    linestyle=style, color=color, label=name,
                    marker='o', markersize=5)
        ax.set_ylabel('重传能耗 (J)')
        ax.set_title('因丢包导致的重传请求能耗 (5分钟平均)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 4. 绘制控制能耗（5分钟间隔）
        ax = axs[3]
        for i, (name, result) in enumerate(self.results.items()):
            style = styles[i % len(styles)]
            color = colors[i % len(colors)]
            ax.plot(result['times'], result['control_energy'],
                    linestyle=style, color=color, label=name,
                    marker='o', markersize=5)
        ax.set_ylabel('控制能耗 (J)')
        ax.set_xlabel('时间 (分钟)')
        ax.set_title('网络探测控制能耗 (5分钟平均)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_individual_algorithm(self, algorithm_name, figsize=(14, 16)):
        """
        为单个算法绘制详细指标图：
        - 完成率：原始决策间隔数据
        - 其他指标：5分钟间隔数据

        Args:
            algorithm_name: 要绘制的算法名称
            figsize: 图形大小，格式为(宽度, 高度)，单位为英寸
        """
        if algorithm_name not in self.results:
            raise ValueError(f"没有算法的结果：{algorithm_name}")

        result = self.results[algorithm_name]

        # 创建带有子图的图形
        fig, axs = plt.subplots(4, 1, figsize=figsize)

        # 设置中文字体（如果可用）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于中文字符
            plt.rcParams['axes.unicode_minus'] = False  # 修复负号"-"显示问题
            plt.rcParams['font.family'] = 'SimHei'
        except:
            pass  # 如果字体不可用，继续使用默认字体

        # 1. 绘制完成率（原始数据点）
        ax = axs[0]
        ax.plot(result['original_times'], [rate * 100 for rate in result['completion_rate']],
                'b-', linewidth=2)
        ax.set_ylabel('完成率 (%)')
        ax.set_title(f'{algorithm_name}: 无重传情况下的数据完成率')
        ax.grid(True, alpha=0.3)

        # 2. 绘制冗余数据量（5分钟间隔）
        ax = axs[1]
        # 转换为KB以便更好地阅读
        redundant_kb = [r / 1024 for r in result['redundant_data']]
        ax.plot(result['times'], redundant_kb, 'ro-', linewidth=2, markersize=6)
        ax.set_ylabel('冗余数据量 (KB)')
        ax.set_title(f'{algorithm_name}: 对解码无增益的冗余数据量 (5分钟平均)')
        ax.grid(True, alpha=0.3)

        # 3. 绘制重传能耗（5分钟间隔）
        ax = axs[2]
        ax.plot(result['times'], result['retransmission_energy'], 'go-', linewidth=2, markersize=6)
        ax.set_ylabel('重传能耗 (J)')
        ax.set_title(f'{algorithm_name}: 因丢包导致的重传请求能耗 (5分钟平均)')
        ax.grid(True, alpha=0.3)

        # 4. 绘制控制能耗（5分钟间隔）
        ax = axs[3]
        ax.plot(result['times'], result['control_energy'], 'mo-', linewidth=2, markersize=6)
        ax.set_ylabel('控制能耗 (J)')
        ax.set_xlabel('时间 (分钟)')
        ax.set_title(f'{algorithm_name}: 网络探测控制能耗 (5分钟平均)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def print_summary(self):
        """
        打印所有算法的汇总统计信息。
        """
        if not self.results:
            raise ValueError("没有结果可总结。请先运行对比。")

        print("\n===== 算法对比总结 =====")

        # 计算平均指标
        for name, result in self.results.items():
            avg_completion_rate = np.mean(result['completion_rate']) * 100
            avg_redundant_data = np.mean(result['redundant_data']) / 1024  # 转换为KB
            avg_retrans_energy = np.mean(result['retransmission_energy'])
            avg_control_energy = np.mean(result['control_energy'])
            total_control_energy = np.sum(result['control_energy'])

            print(f"\n--- {name} ---")
            print(f"平均完成率: {avg_completion_rate:.2f}%")
            print(f"平均冗余数据: {avg_redundant_data:.2f} KB")
            print(f"平均重传能耗: {avg_retrans_energy:.4f} J")
            print(f"平均控制/探测能耗: {avg_control_energy:.4f} J")
            print(f"总控制/探测能耗: {total_control_energy:.4f} J")


# 主函数：运行分析和比较
if __name__ == "__main__":
    comparison = run_full_algorithm_comparison()
    plt.show()