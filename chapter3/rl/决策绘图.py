import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 导入您的路径和决策器类
from paths import Paths
from path import Path
from network_coding_decision import NetworkCodingDecisionMaker


def visualize_network_coding_decisions(maritime_paths: Paths,
                                       duration_minutes: int = 30,
                                       time_step: float = 0.5,
                                       probe_interval: float = 5.0,
                                       figsize: Tuple[int, int] = (14, 10)):
    """
    可视化网络编码决策随时间的变化。

    Args:
        maritime_paths: 包含多条海事通信链路的Paths对象
        duration_minutes: 模拟持续时间（分钟）
        time_step: 时间步长（分钟）
        probe_interval: 网络探测基础间隔（分钟）
        figsize: 图表尺寸
    """
    # 创建决策器实例
    decision_maker = NetworkCodingDecisionMaker()

    # 生成时间点系列
    time_points = np.arange(0, duration_minutes + time_step, time_step)

    # 存储各个时间点的决策结果
    decisions = []

    # 对每个时间点进行决策
    for t in time_points:
        # 调用决策算法
        decision = decision_maker.make_decision(maritime_paths, t, probe_interval)
        decisions.append(decision)

    # 准备绘图数据
    path_ids = maritime_paths.list_paths()
    n_paths = len(path_ids)

    # 创建颜色映射，使用Paths类中已有的颜色
    path_colors = {path_id: maritime_paths.path_colors[path_id] for path_id in path_ids}

    # 提取决策数据
    probe_data = {path_id: [] for path_id in path_ids}

    bandwidth_data = {path_id: [] for path_id in path_ids}
    packet_loss_data = {path_id: [] for path_id in path_ids}
    allocation_data = {path_id: [] for path_id in path_ids}
    coding_rates = []
    decision_values = []

    for i, t in enumerate(time_points):
        decision = decisions[i]

        # 提取编码率
        N, M, R = decision['coding_parameters']
        coding_rate = R / N if N > 0 else 0
        coding_rates.append(coding_rate)

        # 提取决策值
        decision_values.append(decision['decision_value'])

        # 提取每条路径的探测状态、带宽、丢包率和分配数据
        for path_id in path_ids:
            path = maritime_paths.get_path(path_id)

            # 时间索引
            time_idx = int(round(t / path.total_duration * (path.total_points - 1)))

            # 探测状态
            probe_data[path_id].append(decision['probed_paths'].get(path_id, False))

            # 带宽数据
            bandwidth_data[path_id].append(path.bandwidth_series[time_idx])

            # 丢包率数据
            packet_loss_data[path_id].append(path.packet_loss_series[time_idx])

            # 分配数据
            allocation = decision['allocations'].get(path_id, 0)
            allocation_data[path_id].append(allocation)

    # 开始绘图
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)

    # 第一个子图：带宽和分配
    ax1 = fig.add_subplot(gs[0])

    # 绘制带宽曲线及探测标记
    for path_id in path_ids:
        # 绘制带宽曲线
        ax1.plot(time_points, bandwidth_data[path_id],
                 label=f"{path_id} 带宽",
                 color=path_colors[path_id],
                 linestyle='-',
                 linewidth=1.5)

        # 绘制探测标记
        probe_times = [time_points[i] for i in range(len(time_points)) if probe_data[path_id][i]]
        probe_bandwidths = [bandwidth_data[path_id][i] for i in range(len(time_points)) if probe_data[path_id][i]]

        if probe_times:
            ax1.scatter(probe_times, probe_bandwidths,
                        marker='o',
                        s=80,  # 标记大小
                        color=path_colors[path_id],
                        edgecolors='red',
                        linewidths=1.5,
                        alpha=0.7,
                        zorder=10,
                        label=f"{path_id} 探测")

    # 创建第二个y轴用于绘制分配
    ax1_allocation = ax1.twinx()

    # 绘制分配条形图
    bar_width = time_step * 0.8
    for i, path_id in enumerate(path_ids):
        # 计算条形图的偏移，使不同路径的分配条在同一时间点不重叠
        offset = (i - (n_paths - 1) / 2) * (bar_width / n_paths)

        ax1_allocation.bar(time_points + offset, allocation_data[path_id],
                           width=bar_width / n_paths,
                           alpha=0.5,
                           color=path_colors[path_id],
                           label=f"{path_id} 分配")

    # 添加图例和标签
    ax1.set_title('链路带宽、流量分配和网络探测随时间的变化')
    ax1.set_ylabel('带宽 (Mbps)')
    ax1.set_xticklabels([])  # 隐藏x轴刻度标签
    ax1.grid(True, alpha=0.3)

    ax1_allocation.set_ylabel('分配数据包数量')

    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_allocation.get_legend_handles_labels()

    # 过滤重复的探测图例项
    unique_labels = []
    unique_lines = []
    seen_probes = set()

    for line, label in zip(lines1, labels1):
        if "探测" in label:
            path = label.split(" ")[0]
            if path not in seen_probes:
                seen_probes.add(path)
                unique_lines.append(line)
                unique_labels.append(f"网络探测点")
        else:
            unique_lines.append(line)
            unique_labels.append(label)

    ax1.legend(unique_lines + lines2, unique_labels + labels2, loc='upper right')

    # 第二个子图：丢包率和编码率
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # 绘制丢包率曲线和探测标记
    for path_id in path_ids:
        # 丢包率转为百分比
        packet_loss_pct = np.array(packet_loss_data[path_id]) * 100

        # 绘制丢包率曲线
        ax2.plot(time_points, packet_loss_pct,
                 label=f"{path_id} 丢包率",
                 color=path_colors[path_id],
                 linestyle='-',
                 linewidth=1.5)

        # 绘制探测标记
        probe_times = [time_points[i] for i in range(len(time_points)) if probe_data[path_id][i]]
        probe_losses = [packet_loss_pct[i] for i in range(len(time_points)) if probe_data[path_id][i]]

        if probe_times:
            ax2.scatter(probe_times, probe_losses,
                        marker='o',
                        s=80,  # 标记大小
                        color=path_colors[path_id],
                        edgecolors='red',
                        linewidths=1.5,
                        alpha=0.7,
                        zorder=10)

    # 创建第二个y轴用于绘制编码率
    ax2_rate = ax2.twinx()

    # 绘制编码率曲线
    ax2_rate.plot(time_points, coding_rates,
                  label='编码率 (R/N)',
                  color='purple',
                  linestyle='-',
                  linewidth=2.5)

    # 绘制决策值
    ax2_rate.plot(time_points, decision_values,
                  label='决策值',
                  color='navy',
                  linestyle=':',
                  linewidth=2)

    # 添加图例和标签
    ax2.set_title('链路丢包率和编码决策随时间的变化')
    ax2.set_xlabel('时间 (分钟)')
    ax2.set_ylabel('丢包率 (%)')
    ax2.grid(True, alpha=0.3)

    ax2_rate.set_ylabel('编码率 (R/N) 和决策值')

    # 合并两个轴的图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_rate.get_legend_handles_labels()

    # 创建探测标记图例项
    probe_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                             markeredgecolor='red', markersize=10, linewidth=0)

    # 合并图例
    ax2.legend(lines1 + lines2 + [probe_patch],
               labels1 + labels2 + ['网络探测点'],
               loc='upper right')

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为时间标签留出空间

    return fig, decisions


# 示例用法
if __name__ == "__main__":
    # 创建海事链路
    maritime_paths = Paths()

    # 创建第一条海事路径(卫星链路)
    avg_bandwidth_1 = [12, 11, 10, 9, 8, 7, 7, 8, 9, 10, 11, 12]
    avg_packet_loss_1 = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    fluctuation_level_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]

    satellite_path = Path(
        avg_bandwidth_series=avg_bandwidth_1,
        avg_packet_loss_series=avg_packet_loss_1,
        fluctuation_level_series=fluctuation_level_1,
        time_slot_duration=5,  # 每时隙5分钟
        points_per_slot=300  # 每时隙300个点(每秒1个点)
    )

    # 创建第二条海事路径(沿海无线电)
    avg_bandwidth_2 = [8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6]
    avg_packet_loss_2 = [0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    fluctuation_level_2 = [3, 3, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4]

    coastal_radio_path = Path(
        avg_bandwidth_series=avg_bandwidth_2,
        avg_packet_loss_series=avg_packet_loss_2,
        fluctuation_level_series=fluctuation_level_2,
        time_slot_duration=5,
        points_per_slot=300
    )

    # 创建第三条海事路径(HF链接)，具有更不稳定的特性
    avg_bandwidth_3 = [15, 8, 12, 5, 10, 15, 8, 12, 5, 10, 15, 8]
    avg_packet_loss_3 = [0.02, 0.09, 0.05, 0.12, 0.06, 0.02, 0.09, 0.05, 0.12, 0.06, 0.02, 0.09]
    fluctuation_level_3 = [5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4]

    hf_link_path = Path(
        avg_bandwidth_series=avg_bandwidth_3,
        avg_packet_loss_series=avg_packet_loss_3,
        fluctuation_level_series=fluctuation_level_3,
        time_slot_duration=5,
        points_per_slot=300
    )

    # 将海事路径添加到容器
    maritime_paths.add_path("path1", satellite_path, color="#1f77b4")
    maritime_paths.add_path("path2", coastal_radio_path, color="#ff7f0e")
    maritime_paths.add_path("HF链接", hf_link_path, color="#2ca02c")

    # 可视化30分钟内的网络编码决策
    fig, decisions = visualize_network_coding_decisions(
        maritime_paths=maritime_paths,
        duration_minutes=30,
        time_step=0.5,  # 每0.5分钟一个决策点
        probe_interval=5.0  # 基础探测间隔为5分钟
    )

    # 保存图片
    plt.savefig('network_coding_decisions.png', dpi=300, bbox_inches='tight')

    # 展示图表
    plt.show()

    # 打印部分决策结果
    print("\n网络编码决策结果示例:")
    print("=" * 80)
    for i in [0, 10, 20, 40, 60]:  # 选取几个时间点
        if i < len(decisions):
            decision = decisions[i]
            time_point = decision['time_point']
            N, M, R = decision['coding_parameters']
            coding_rate = R / N if N > 0 else 0

            print(f"时间: {time_point:.1f} 分钟")
            print(f"  丢包率: " + ", ".join(
                [f"{path}: {loss * 100:.2f}%" for path, loss in decision['packet_loss_rates'].items()]))
            print(f"  编码参数: N={N}, M={M}, R={R} (编码率: {coding_rate:.3f})")
            print(f"  分配情况: " + ", ".join([f"{path}: {alloc}" for path, alloc in decision['allocations'].items()]))
            print(f"  网络探测: " + ", ".join(
                [f"{path}: {'是' if probed else '否'}" for path, probed in decision['probed_paths'].items()]))
            print(f"  决策值: {decision['decision_value']:.4f}")
            print("-" * 80)
