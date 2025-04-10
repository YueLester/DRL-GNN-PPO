
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from path import Path
import time
from typing import List, Dict, Tuple
import math
from matplotlib.gridspec import GridSpec


class NetworkTransmissionSimulator:
    """
    Simulates network transmission using two paths with network coding.

    Simulates a decision process at each time slot (e.g. every 10 seconds), choosing:
    - Whether to probe each path to get updated network conditions
    - How many data packets (N) and redundancy packets (R) to send
    - How to distribute packets between paths (w1, w2) where w1 + w2 = N + R

    Attributes:
        path1: First communication path
        path2: Second communication path
        decision_interval: Interval between decisions in seconds
        simulation_duration: Total simulation duration in minutes
        max_packets: Maximum number of data packets (N)
        max_redundancy: Maximum number of redundancy packets (R)
        probe_interval: Minimum interval between probes in seconds
        last_probe_time1: Last time path1 was probed
        last_probe_time2: Last time path2 was probed
        probe_history: History of probe decisions
        transmission_history: History of transmission decisions
    """

    def __init__(self, path1: Path, path2: Path, decision_interval: int = 10,
                 simulation_duration: int = 60, max_packets: int = 64,
                 max_redundancy: int = 24, probe_interval: int = 30):
        """
        Initialize the network transmission simulator.

        Args:
            path1: First communication path
            path2: Second communication path
            decision_interval: Interval between decisions in seconds
            simulation_duration: Total simulation duration in minutes
            max_packets: Maximum number of data packets
            max_redundancy: Maximum number of redundancy packets
            probe_interval: Minimum interval between probes in seconds
        """
        self.path1 = path1
        self.path2 = path2
        self.decision_interval = decision_interval  # seconds
        self.simulation_duration = simulation_duration  # minutes
        self.max_packets = max_packets
        self.max_redundancy = max_redundancy
        self.probe_interval = probe_interval  # seconds

        # Points per minute for each path
        self.points_per_minute1 = path1.total_points / path1.total_duration
        self.points_per_minute2 = path2.total_points / path2.total_duration

        # Track last probe times
        self.last_probe_time1 = -float('inf')
        self.last_probe_time2 = -float('inf')

        # History of decisions and observations
        self.probe_history = {'times': [], 'path1': [], 'path2': []}
        self.transmission_history = {
            'times': [],
            'N': [],
            'R': [],
            'w1': [],
            'w2': [],
            'bandwidth1': [],
            'bandwidth2': [],
            'packet_loss1': [],
            'packet_loss2': [],
            'success_rate': [],
            'redundancy_ratio': []
        }

        # Current estimates of path parameters (to be updated by probes)
        self.current_bandwidth1 = None
        self.current_packet_loss1 = None
        self.current_bandwidth2 = None
        self.current_packet_loss2 = None

        # Previous estimates for calculating change rates
        self.prev_bandwidth1 = None
        self.prev_packet_loss1 = None
        self.prev_bandwidth2 = None
        self.prev_packet_loss2 = None

        # Time since last probe for each path (in seconds)
        self.time_since_probe1 = float('inf')
        self.time_since_probe2 = float('inf')

    def _decide_probe(self, current_time: float, path_id: int) -> bool:
        """
        决定是否对路径进行探测，根据网络波动情况决定。

        Args:
            current_time: 当前模拟时间（秒）
            path_id: 路径标识符（1或2）

        Returns:
            布尔值，表示是否要进行探测
        """
        # 初始化探测（第一次决策点）
        if current_time == 0:
            return True

        # 获取上次探测时间和计算经过时间
        if path_id == 1:
            last_probe_time = self.last_probe_time1
            time_since_probe = current_time - last_probe_time

            # 获取上次和当前的路径状态估计
            current_bandwidth = self.current_bandwidth1
            current_packet_loss = self.current_packet_loss1

            # 获取真实的当前路径状态（用于判断是否需要探测）
            actual_state = self._get_path_state(current_time, self.path1)
            actual_bandwidth = actual_state['bandwidth']
            actual_packet_loss = actual_state['packet_loss']

            # 计算估计值与实际值的偏差
            if current_bandwidth is not None and current_packet_loss is not None:
                bandwidth_error = abs(current_bandwidth - actual_bandwidth)
                packet_loss_error = abs(current_packet_loss - actual_packet_loss)
            else:
                # 如果没有当前估计，则需要探测
                return True
        else:  # path_id == 2
            last_probe_time = self.last_probe_time2
            time_since_probe = current_time - last_probe_time

            # 获取上次和当前的路径状态估计
            current_bandwidth = self.current_bandwidth2
            current_packet_loss = self.current_packet_loss2

            # 获取真实的当前路径状态（用于判断是否需要探测）
            actual_state = self._get_path_state(current_time, self.path2)
            actual_bandwidth = actual_state['bandwidth']
            actual_packet_loss = actual_state['packet_loss']

            # 计算估计值与实际值的偏差
            if current_bandwidth is not None and current_packet_loss is not None:
                bandwidth_error = abs(current_bandwidth - actual_bandwidth)
                packet_loss_error = abs(current_packet_loss - actual_packet_loss)
            else:
                # 如果没有当前估计，则需要探测
                return True

        # 定义触发探测的阈值
        bandwidth_threshold = 1.0  # Mbps，带宽变化阈值
        packet_loss_threshold = 0.02  # 2%，丢包率变化阈值

        # 基于网络变化决定是否探测
        if bandwidth_error > bandwidth_threshold or packet_loss_error > packet_loss_threshold:
            # 如果偏差超过阈值，则需要探测
            return True
        elif time_since_probe >= 5 * self.probe_interval:
            # 如果长时间未探测（超过正常间隔的5倍），强制探测
            return True
        elif time_since_probe >= self.probe_interval:
            # 如果已经超过了最小探测间隔但网络稳定，以较低概率探测
            # 随着未探测时间增加，探测概率逐渐提高
            probe_probability = min(0.3, (time_since_probe - self.probe_interval) / (4 * self.probe_interval))
            return np.random.random() < probe_probability

        # 其他情况下不探测
        return False

    def _get_path_state(self, current_time: float, path: Path) -> Dict:
        """
        Get the current state of a path at the given time.

        Args:
            current_time: Current simulation time in seconds
            path: The path object

        Returns:
            Dictionary with path state (bandwidth, packet_loss)
        """
        # Convert time from seconds to minutes for path functions
        time_in_minutes = current_time / 60.0

        # Get the state
        state = path.get_state_at_time(time_in_minutes)

        return {
            'bandwidth': state['bandwidth'],
            'packet_loss': state['packet_loss']
        }

    def _update_path_estimates(self, current_time: float, probe_path1: bool, probe_path2: bool):
        """
        Update the path estimates based on probing decisions.

        Args:
            current_time: Current simulation time in seconds
            probe_path1: Whether to probe path 1
            probe_path2: Whether to probe path 2
        """
        # Initialize estimates if this is the first update
        if self.current_bandwidth1 is None or self.current_packet_loss1 is None:
            state1 = self._get_path_state(current_time, self.path1)
            self.current_bandwidth1 = state1['bandwidth']
            self.current_packet_loss1 = state1['packet_loss']
            self.prev_bandwidth1 = self.current_bandwidth1
            self.prev_packet_loss1 = self.current_packet_loss1

        if self.current_bandwidth2 is None or self.current_packet_loss2 is None:
            state2 = self._get_path_state(current_time, self.path2)
            self.current_bandwidth2 = state2['bandwidth']
            self.current_packet_loss2 = state2['packet_loss']
            self.prev_bandwidth2 = self.current_bandwidth2
            self.prev_packet_loss2 = self.current_packet_loss2

        # Save previous values for change rate calculation
        if probe_path1 or probe_path2:
            # Only save previous values when we're about to update with new values
            if probe_path1:
                self.prev_bandwidth1 = self.current_bandwidth1
                self.prev_packet_loss1 = self.current_packet_loss1

            if probe_path2:
                self.prev_bandwidth2 = self.current_bandwidth2
                self.prev_packet_loss2 = self.current_packet_loss2

        # Update estimates for path 1
        if probe_path1:
            state1 = self._get_path_state(current_time, self.path1)
            self.current_bandwidth1 = state1['bandwidth']
            self.current_packet_loss1 = state1['packet_loss']
            self.last_probe_time1 = current_time
            self.time_since_probe1 = 0
        else:
            self.time_since_probe1 = current_time - self.last_probe_time1

        # Update estimates for path 2
        if probe_path2:
            state2 = self._get_path_state(current_time, self.path2)
            self.current_bandwidth2 = state2['bandwidth']
            self.current_packet_loss2 = state2['packet_loss']
            self.last_probe_time2 = current_time
            self.time_since_probe2 = 0
        else:
            self.time_since_probe2 = current_time - self.last_probe_time2

    def _calculate_min_redundancy(self, N: int, packet_loss1: float, packet_loss2: float, w1_ratio: float) -> int:
        """
        基于丢包率计算所需的最小冗余包数量。

        Args:
            N: 数据包数量
            packet_loss1: 路径1的丢包率
            packet_loss2: 路径2的丢包率
            w1_ratio: 在路径1上传输的数据包比例

        Returns:
            所需的最小冗余包数量
        """
        # 计算每条路径上的预期丢包
        expected_loss1 = w1_ratio * packet_loss1
        expected_loss2 = (1 - w1_ratio) * packet_loss2

        # 总体预期丢包率
        combined_loss_rate = expected_loss1 + expected_loss2

        # 计算所需的最小冗余（预期丢包加安全边际）
        # 我们至少需要足够的冗余来覆盖预期的丢包
        min_redundancy = math.ceil(N * combined_loss_rate * 1.3)  # 增加30%的安全边际

        return min_redundancy

    def _calculate_packet_distribution(self, N: int, R: int, bandwidth1: float, bandwidth2: float) -> Tuple[int, int]:
        """
        计算数据包在两条路径上的分配。

        Args:
            N: 数据包数量
            R: 冗余包数量
            bandwidth1: 路径1的带宽
            bandwidth2: 路径2的带宽

        Returns:
            数据包分配的元组 (w1, w2)
        """
        # 计算总可用带宽
        total_bandwidth = bandwidth1 + bandwidth2

        # 基于带宽确定分配比例
        if total_bandwidth > 0:
            w1_ratio = bandwidth1 / total_bandwidth
        else:
            # 如果总带宽为0（两条路径都不可用），使用均等分配
            w1_ratio = 0.5

        # 计算数据包分配 (w1, w2)
        total_packets = N + R
        w1 = int(round(total_packets * w1_ratio))
        w2 = total_packets - w1

        # 确保每条有效路径上至少有一些数据包
        if w1 == 0 and bandwidth1 > 0:
            w1 = 1
            w2 = total_packets - 1
        elif w2 == 0 and bandwidth2 > 0:
            w2 = 1
            w1 = total_packets - 1

        return w1, w2

    def _calculate_redundancy_for_uncertainty(self, min_redundancy: int, time_since_probe1: float,
                                              time_since_probe2: float) -> int:
        """
        基于上次探测后的时间（不确定性）计算额外冗余。

        Args:
            min_redundancy: 基于丢包率计算的基本冗余
            time_since_probe1: 自上次对路径1探测以来的时间
            time_since_probe2: 自上次对路径2探测以来的时间

        Returns:
            包括不确定性因素的总冗余
        """
        # 更高的不确定性（自上次探测的时间）导致更多的冗余
        time_factor1 = min(1.0, time_since_probe1 / (5 * self.probe_interval))
        time_factor2 = min(1.0, time_since_probe2 / (5 * self.probe_interval))
        time_factor = max(time_factor1, time_factor2)

        # 为不确定性添加额外冗余
        # 随着时间增加，增加额外冗余的比例
        extra_redundancy = int(self.max_redundancy * 0.4 * time_factor)

        # 总冗余，不超过最大冗余限制
        total_redundancy = min(self.max_redundancy, min_redundancy + extra_redundancy)

        return total_redundancy

    def _decide_transmission(self, current_time: float) -> Dict:
        """
        基于当前路径状态估计决定传输参数。

        此函数确定：
        1. 数据包数量 (N)
        2. 冗余包数量 (R)
        3. 两条路径上的数据包分配 (w1, w2)

        Args:
            current_time: 当前模拟时间（秒）

        Returns:
            包含传输参数的字典 (N, R, w1, w2)
        """
        # ---- 1. 获取当前路径状态（模拟目的） ----
        actual_state1 = self._get_path_state(current_time, self.path1)
        actual_state2 = self._get_path_state(current_time, self.path2)

        # ---- 2. 设置数据包数量 (N) ----
        N = self.max_packets  # 使用最大允许的数据包数量

        # ---- 3. 计算用于数据包分配的带宽比例 ----
        total_bandwidth = self.current_bandwidth1 + self.current_bandwidth2
        if total_bandwidth > 0:
            w1_ratio = self.current_bandwidth1 / total_bandwidth
        else:
            w1_ratio = 0.5

        # ---- 4. 基于估计的丢包率计算最小冗余 ----
        min_redundancy = self._calculate_min_redundancy(
            N,
            self.current_packet_loss1,
            self.current_packet_loss2,
            w1_ratio
        )

        # ---- 5. 增加额外冗余以应对不确定性 ----
        R = self._calculate_redundancy_for_uncertainty(
            min_redundancy,
            self.time_since_probe1,
            self.time_since_probe2
        )

        # ---- 6. 计算数据包在路径上的分配 ----
        w1, w2 = self._calculate_packet_distribution(
            N, R, self.current_bandwidth1, self.current_bandwidth2
        )

        # ---- 7. 计算预期成功率 ----
        # 基于实际路径状态（在真实系统中，这只能在传输后获知）
        success_rate = self._calculate_success_rate(
            N, R, w1, w2,
            actual_state1['packet_loss'],
            actual_state2['packet_loss']
        )

        # ---- 8. 计算冗余率 ----
        redundancy_ratio = R / (N + R) if (N + R) > 0 else 0

        # ---- 9. 返回完整传输计划 ----
        return {
            'N': N,  # 数据包数量
            'R': R,  # 冗余包数量
            'w1': w1,  # 分配给路径1的包数
            'w2': w2,  # 分配给路径2的包数
            'bandwidth1': actual_state1['bandwidth'],  # 路径1的实际带宽
            'bandwidth2': actual_state2['bandwidth'],  # 路径2的实际带宽
            'packet_loss1': actual_state1['packet_loss'],  # 路径1的实际丢包率
            'packet_loss2': actual_state2['packet_loss'],  # 路径2的实际丢包率
            'success_rate': success_rate,  # 预期成功率
            'redundancy_ratio': redundancy_ratio  # 冗余率 R/(N+R)
        }

    def _nCr(self, n: int, k: int) -> float:
        """
        计算组合数 C(n,k) = n! / (k! * (n-k)!)
        这是math.comb()的替代函数，适用于Python 3.8以下版本

        Args:
            n: 总元素数
            k: 选择的元素数

        Returns:
            组合数C(n,k)
        """
        # 如果k大于n，返回0
        if k > n:
            return 0
        # 使用对称性优化计算
        if k > n - k:
            k = n - k
        # 计算组合数
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    def _calculate_success_rate(self, N: int, R: int, w1: int, w2: int,
                                packet_loss1: float, packet_loss2: float) -> float:
        """
        计算传输的预期成功率。

        在网络编码中，我们需要在N+R个总包中接收至少N个包才能成功解码。

        Args:
            N: 数据包数量
            R: 冗余包数量
            w1: 通过路径1发送的包数
            w2: 通过路径2发送的包数
            packet_loss1: 路径1的丢包率
            packet_loss2: 路径2的丢包率

        Returns:
            预期成功率（0-1）
        """
        # 使用二项式分布计算从路径1和路径2接收包的概率分布

        # 从路径1成功接收的包的概率分布
        p1_success = 1 - packet_loss1
        received_packets_prob1 = [0] * (w1 + 1)
        for i in range(w1 + 1):
            # 使用二项式概率公式：P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
            received_packets_prob1[i] = self._nCr(w1, i) * (p1_success ** i) * ((1 - p1_success) ** (w1 - i))

        # 从路径2成功接收的包的概率分布
        p2_success = 1 - packet_loss2
        received_packets_prob2 = [0] * (w2 + 1)
        for i in range(w2 + 1):
            received_packets_prob2[i] = self._nCr(w2, i) * (p2_success ** i) * ((1 - p2_success) ** (w2 - i))

        # 计算总共接收到至少N个包的概率（成功率）
        success_prob = 0.0

        # 对于路径1上接收到的每种可能的包数量
        for i in range(w1 + 1):
            # 要使总接收包数 >= N，需要从路径2接收 >= (N-i) 个包
            needed_from_path2 = max(0, N - i)

            # 如果从路径2需要的包超过了w2，则此情况无法成功
            if needed_from_path2 > w2:
                continue

            # 计算从路径2接收足够包的概率
            prob_enough_from_path2 = 0.0
            for j in range(needed_from_path2, w2 + 1):
                prob_enough_from_path2 += received_packets_prob2[j]

            # 将此情况的概率添加到总成功概率
            success_prob += received_packets_prob1[i] * prob_enough_from_path2

        return success_prob

    def simulate(self):
        """Run the complete network transmission simulation."""
        # Convert simulation duration from minutes to seconds
        duration_seconds = self.simulation_duration * 60

        # Number of decision points
        num_decisions = int(duration_seconds / self.decision_interval)

        # Run simulation
        for i in range(num_decisions):
            current_time = i * self.decision_interval

            # Decide whether to probe each path
            probe_path1 = self._decide_probe(current_time, 1)
            probe_path2 = self._decide_probe(current_time, 2)

            # Update path estimates based on probes
            self._update_path_estimates(current_time, probe_path1, probe_path2)

            # Record probe decisions
            self.probe_history['times'].append(current_time)
            self.probe_history['path1'].append(1 if probe_path1 else 0)
            self.probe_history['path2'].append(1 if probe_path2 else 0)

            # Decide transmission parameters
            tx_params = self._decide_transmission(current_time)

            # Record transmission decisions
            self.transmission_history['times'].append(current_time)
            self.transmission_history['N'].append(tx_params['N'])
            self.transmission_history['R'].append(tx_params['R'])
            self.transmission_history['w1'].append(tx_params['w1'])
            self.transmission_history['w2'].append(tx_params['w2'])
            self.transmission_history['bandwidth1'].append(tx_params['bandwidth1'])
            self.transmission_history['bandwidth2'].append(tx_params['bandwidth2'])
            self.transmission_history['packet_loss1'].append(tx_params['packet_loss1'])
            self.transmission_history['packet_loss2'].append(tx_params['packet_loss2'])
            self.transmission_history['success_rate'].append(tx_params['success_rate'])
            self.transmission_history['redundancy_ratio'].append(tx_params['redundancy_ratio'])

    def plot_results(self, figsize=(15, 12)):
        """
        Plot the simulation results.

        Args:
            figsize: Figure size as (width, height) in inches
        """
        # Convert times from seconds to minutes for plotting
        times_minutes = np.array(self.transmission_history['times']) / 60.0

        # 全局设置字体大小
        plt.rcParams['font.size'] = 14

        # Create figure with GridSpec for flexible layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 1, height_ratios=[4, 4, 2, 1])
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False

        # Plot 1: Bandwidth and allocation (w1, w2)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(times_minutes, self.transmission_history['bandwidth1'], 'b-', label='路径1带宽')
        ax1.plot(times_minutes, self.transmission_history['bandwidth2'], 'r-', label='路径1带宽')

        for i, time in enumerate(self.probe_history['times']):
            time_min = time / 60.0
            if self.probe_history['path1'][i]:
                ax1.plot(time_min, 0.5, 'bv', markersize=8, alpha=0.7)
            if self.probe_history['path2'][i]:
                ax1.plot(time_min, 0, 'r^', markersize=8, alpha=0.7)


        ax1.set_ylabel('带宽 (Mbps)', color='black')
        ax1.set_xlabel('时间 (min)', color='black')

        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)

        for i in range(len(self.transmission_history['w1'])):
            temp = self.transmission_history['w1'][i] + self.transmission_history['w2'][i]
            self.transmission_history['w1'][i] = self.transmission_history['w1'][i] / temp
            self.transmission_history['w2'][i] = self.transmission_history['w2'][i] / temp
            self.transmission_history['R'][i] = self.transmission_history['R'][i] / temp


        # Create second y-axis for packet allocation
        ax1_right = ax1.twinx()
        ax1_right.plot(times_minutes, self.transmission_history['w1'], 'b--', label='路径1流量比例', alpha=0.7)
        ax1_right.plot(times_minutes, self.transmission_history['w2'], 'r--', label='路径2流量比例', alpha=0.7)
        print(type(self.transmission_history['w1']))
        ax1_right.set_ylabel('流量分配比例', color='gray')
        ax1_right.tick_params(axis='y', labelcolor='gray')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_right.get_legend_handles_labels()

        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='v', color='b', linestyle='None', markersize=8, alpha=0.7),
            Line2D([0], [0], marker='^', color='r', linestyle='None', markersize=8, alpha=0.7),
        ]
        ax1.legend(lines1 + lines2 + custom_lines, labels1 + labels2 + ['路径1探测', '路径2探测'], loc='upper right')

        ax1.set_title('网络带宽，流量分配决策，网络探测决策 vs. 时间图')



        # Plot 2: Packet Loss and Redundancy (R)
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(times_minutes, [pl * 100 for pl in self.transmission_history['packet_loss1']], 'b-',
                 label='路径1丢包率 (%)')
        ax2.plot(times_minutes, [pl * 100 for pl in self.transmission_history['packet_loss2']], 'r-',
                 label='路径2丢包率 (%)')
        ax2.set_ylabel('丢包率 (%)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('时间 (min)', color='black')

        # Create second y-axis for redundancy
        ax2_right = ax2.twinx()
        ax2_right.plot(times_minutes, self.transmission_history['R'], 'g-', label='传输冗余率 （R/M）', alpha=0.7)
        ax2_right.set_ylabel('传输冗余率 ', color='green')
        ax2_right.set_ylim(0,0.3)
        ax2_right.tick_params(axis='y', labelcolor='green')

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax2.set_title('网络丢包率，冗余决策 vs. 时间')

        # Plot 3: N and R values
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(times_minutes, self.transmission_history['N'], 'b-', label='Data Packets (N)')
        ax3.plot(times_minutes, self.transmission_history['R'], 'r-', label='Redundancy Packets (R)')
        ax3.set_ylabel('Number of Packets', color='black')
        ax3.tick_params(axis='y', labelcolor='black')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        ax3.set_title('Data and Redundancy Packets Over Time')

        # Plot 4: Success Rate and Probing Events
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(times_minutes, [sr * 100 for sr in self.transmission_history['success_rate']], 'g-',
                 label='Success Rate (%)')
        ax4.set_ylabel('Success Rate (%)', color='black')
        ax4.set_ylim(0, 105)  # Leave room for probe markers
        ax4.tick_params(axis='y', labelcolor='black')
        ax4.grid(True, alpha=0.3)

        # Add probe markers
        for i, time in enumerate(self.probe_history['times']):
            time_min = time / 60.0
            if self.probe_history['path1'][i]:
                ax4.plot(time_min, 102, 'bv', markersize=8, alpha=0.7)
            if self.probe_history['path2'][i]:
                ax4.plot(time_min, 98, 'r^', markersize=8, alpha=0.7)

        # Add legend for probes
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='v', color='b', linestyle='None', markersize=8, alpha=0.7),
            Line2D([0], [0], marker='^', color='r', linestyle='None', markersize=8, alpha=0.7),
            Line2D([0], [0], color='g', lw=2)
        ]
        ax4.legend(custom_lines, ['Path 1 Probe', 'Path 2 Probe', 'Success Rate'], loc='upper right')

        ax4.set_title('Success Rate and Probing Events')
        ax4.set_xlabel('Time (minutes)')

        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Create two paths with different characteristics
    # Path 1: More stable, higher bandwidth, lower packet loss
    # Path 2: Less stable, lower bandwidth, higher packet loss

    # 60 minutes simulation with 12 time slots (5 min each)
    total_slots = 6

    # Path 1 parameters (more stable path)
    bw1_avg = [12, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 11]
    pl1_avg = [0.03, 0.04, 0.05, 0.04, 0.03, 0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.04]
    fluct1_level = [1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2]  # Lower fluctuation

    # Path 2 parameters (less stable path)
    bw2_avg = [8, 7, 6, 5, 6, 7, 8, 9, 8, 7, 6, 7]
    pl2_avg = [0.08, 0.09, 0.10, 0.12, 0.10, 0.09, 0.08, 0.07, 0.08, 0.09, 0.10, 0.09]
    fluct2_level = [3, 4, 5, 4, 3, 4, 3, 3, 4, 5, 4, 3]  # Higher fluctuation

    # Create paths with different fluctuation patterns
    path1 = Path(
        avg_bandwidth_series=bw1_avg,
        avg_packet_loss_series=pl1_avg,
        fluctuation_level_series=fluct1_level,
        time_slot_duration=5,  # 5 minutes per slot
        points_per_slot=300,  # 300 points per slot (1 point per second)
        fluctuation_pattern='sine'  # Use sine wave pattern for path 1
    )

    path2 = Path(
        avg_bandwidth_series=bw2_avg,
        avg_packet_loss_series=pl2_avg,
        fluctuation_level_series=fluct2_level,
        time_slot_duration=5,  # 5 minutes per slot
        points_per_slot=300,  # 300 points per slot (1 point per second)
        fluctuation_pattern='mixed'  # Use mixed pattern for path 2
    )

    # Create the simulator
    simulator = NetworkTransmissionSimulator(
        path1=path1,
        path2=path2,
        decision_interval=10,  # Make decisions every 10 seconds
        simulation_duration=40,  # 60 minutes total simulation
        max_packets=64,  # Max data packets (N)
        max_redundancy=24,  # Max redundancy packets (R)
        probe_interval=30  # Min interval between probes (seconds)
    )

    # Run the simulation
    print("Running network transmission simulation...")
    simulator.simulate()

    # Plot the results
    print("Plotting results...")
    fig = simulator.plot_results()
    plt.savefig('network_transmission_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\nSimulation Summary:")
    print("Total simulation time: {0} minutes".format(simulator.simulation_duration))
    print("Decision intervals: {0} seconds".format(simulator.decision_interval))
    print("Total decisions made: {0}".format(len(simulator.transmission_history['times'])))

    # Calculate averages
    avg_N = np.mean(simulator.transmission_history['N'])
    avg_R = np.mean(simulator.transmission_history['R'])
    avg_w1 = np.mean(simulator.transmission_history['w1'])
    avg_w2 = np.mean(simulator.transmission_history['w2'])
    avg_success = np.mean(simulator.transmission_history['success_rate']) * 100
    avg_redundancy_ratio = np.mean(simulator.transmission_history['redundancy_ratio'])

    print("Average data packets (N): {0:.2f}".format(avg_N))
    print("Average redundancy packets (R): {0:.2f}".format(avg_R))
    print("Average redundancy ratio (R/(N+R)): {0:.4f}".format(avg_redundancy_ratio))
    print("Average path 1 allocation (w1): {0:.2f}".format(avg_w1))
    print("Average path 2 allocation (w2): {0:.2f}".format(avg_w2))
    print("Average success rate: {0:.2f}%".format(avg_success))

    # Count probes
    total_probes1 = sum(simulator.probe_history['path1'])
    total_probes2 = sum(simulator.probe_history['path2'])
    print("Total probes on path 1: {0}".format(total_probes1))
    print("Total probes on path 2: {0}".format(total_probes2))