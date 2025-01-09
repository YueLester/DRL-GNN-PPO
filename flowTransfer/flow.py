import pickle
import random

import gym
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

import math


def free_space_path_loss(distance, frequency):
    fspl = 20 * math.log10(distance) + 20 * math.log10(frequency) - 147.55
    return fspl

def received_power(Pt, distance, frequency):
    fspl_db = free_space_path_loss(distance, frequency)
    Pr = Pt - 10**(fspl_db / 10)  # 将 dB 转换为瓦特
    return Pr

def shannon_hartley_bandwidth(Pr, B, N0):
    if Pr <= 0 or N0 <= 0:
        return 0
    C = B * math.log2(1 + Pr / (N0 * B))
    return C

def free_space_path_loss(distance, frequency):
    """
    计算自由空间路径损耗

    参数:
    distance: 通信距离 (单位：米)
    frequency: 载频 (单位：赫兹)

    返回:
    fspl: 自由空间路径损耗 (单位：dB)
    """
    speed_of_light = 3e8  # 光速 (单位：米/秒)
    fspl = 20 * math.log10(distance) + 20 * math.log10(frequency) - 147.55
    return fspl


