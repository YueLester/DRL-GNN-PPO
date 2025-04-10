import numpy as np
import matplotlib.pyplot as plt
from path import Path
from trans1 import NetworkTransmissionSimulator
from final_compare import (
    AMRBECSimulator,
    AAEEMRSimulator,
    RMPRSimulator,
    AFBGPSRSimulator,
    AlgorithmComparison,
    run_full_algorithm_comparison
)


def run_analysis():
    """
    运行详细的海上网络传输算法分析。
    比较不同传输算法在时变环境下的性能差异。
    """
    print("开始网络算法分析...")

    # 运行完整的算法对比分析
    comparison = run_full_algorithm_comparison()

    # 显示图形（如果在交互环境中）
    plt.show()

    return comparison


if __name__ == "__main__":
    comparison = run_analysis()