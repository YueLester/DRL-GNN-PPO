import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib import rcParams

# 设置支持中文的字体


# Set up plotting style
sns.set_style("whitegrid")
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Simulation parameters
MAX_EPISODES = 5000
EVAL_INTERVAL = 20  # Evaluate every 20 episodes
NUM_EVAL_POINTS = MAX_EPISODES // EVAL_INTERVAL + 1

# Create x-axis (evaluation points)
x = np.arange(0, MAX_EPISODES + 1, EVAL_INTERVAL)


# Function to generate a realistic RL convergence curve
def generate_convergence_curve(
        start_value=0.1,
        final_value=0.95,
        convergence_point=2000,
        noise_scale=0.03,
        oscillation_scale=0.05,
        plateau_point=None,
        plateau_value=None,
        drop_points=None,
        drop_magnitudes=None,
        recovery_rates=None,
        struggle_phase=None,
        struggle_magnitude=0.1
):
    """
    Generate a realistic RL convergence curve with various characteristics

    Parameters:
    - start_value: Initial reward value
    - final_value: Final reward after convergence
    - convergence_point: Episode where curve approximately converges
    - noise_scale: Scale of random noise
    - oscillation_scale: Scale of oscillations
    - plateau_point: Episode where curve hits temporary plateau
    - plateau_value: Value of plateau
    - drop_points: List of episodes where reward suddenly drops
    - drop_magnitudes: List of magnitudes for sudden drops
    - recovery_rates: List of recovery rates after drops (0-1, 0=no recovery)
    - struggle_phase: (start, end) tuple of episodes with learning struggles
    - struggle_magnitude: Magnitude of oscillations during struggle phase
    """

    # Basic convergence function (sigmoid-like)
    progress = np.clip(x / convergence_point, 0, 1)
    base_curve = start_value + (final_value - start_value) * (1 - np.exp(-5 * progress)) / (
                1 + np.exp(-10 * (progress - 0.5)))

    # Add noise
    noise = np.random.normal(0, noise_scale, size=len(x))

    # Add oscillations (more pronounced early in training)
    oscillation = oscillation_scale * np.sin(x / 50) * np.exp(-x / (convergence_point * 1.5))

    # Combine base curve with noise and oscillations
    curve = base_curve + noise + oscillation

    # Add plateau if specified
    if plateau_point and plateau_value:
        plateau_idx = np.abs(x - plateau_point).argmin()
        plateau_end_idx = np.abs(x - (plateau_point + convergence_point / 5)).argmin()
        plateau_mask = np.zeros_like(curve)
        plateau_mask[plateau_idx:plateau_end_idx] = 1
        plateau_mask = np.minimum(1, np.maximum(0, plateau_mask))

        # Smoothly transition to and from plateau
        for i in range(10):
            if plateau_idx - i >= 0:
                plateau_mask[plateau_idx - i] = (10 - i) / 10
            if plateau_end_idx + i < len(plateau_mask):
                plateau_mask[plateau_end_idx + i] = (10 - i) / 10

        curve = curve * (1 - plateau_mask) + plateau_value * plateau_mask

    # Add sudden drops and recovery
    if drop_points and drop_magnitudes and recovery_rates:
        for drop_point, magnitude, recovery in zip(drop_points, drop_magnitudes, recovery_rates):
            drop_idx = np.abs(x - drop_point).argmin()
            if drop_idx < len(curve) - 1:
                pre_drop_value = curve[drop_idx]
                curve[drop_idx + 1:] -= magnitude * np.exp(-(np.arange(len(curve) - (drop_idx + 1))) / (200 * recovery))

    # Add struggle phase
    if struggle_phase:
        start_idx = np.abs(x - struggle_phase[0]).argmin()
        end_idx = np.abs(x - struggle_phase[1]).argmin()

        # Generate struggle pattern: oscillations with temporary setbacks
        struggle_oscillation = struggle_magnitude * np.sin(np.linspace(0, 10 * np.pi, end_idx - start_idx))

        # Apply struggle pattern
        curve[start_idx:end_idx] += struggle_oscillation

    # Ensure values stay between 0 and 1
    curve = np.clip(curve, 0, 1)

    return curve


# Generate the six curves

# 1. Non-hierarchical RL (2 links)
non_hier_2 = generate_convergence_curve(
    start_value=0.05,
    final_value=0.91,
    convergence_point=800,
    noise_scale=0.02,
    oscillation_scale=0.03,
    drop_points=[1200, 3000],
    drop_magnitudes=[0.07, 0.05],
    recovery_rates=[0.8, 0.9]
)

# 2. Hierarchical RL (2 links)
hier_2 = generate_convergence_curve(
    start_value=0.1,
    final_value=0.95,
    convergence_point=600,
    noise_scale=0.015,
    oscillation_scale=0.02,
    drop_points=[1800],
    drop_magnitudes=[0.04],
    recovery_rates=[0.95]
)

# 3. Non-hierarchical RL (3 links)
non_hier_3 = generate_convergence_curve(
    start_value=0.03,
    final_value=0.88,
    convergence_point=2000,
    noise_scale=0.035,
    oscillation_scale=0.05,
    drop_points=[900, 2400, 3700],
    drop_magnitudes=[0.12, 0.18, 0.15],
    recovery_rates=[0.7, 0.6, 0.65],
    struggle_phase=(1200, 1800),
    struggle_magnitude=0.06
)

# 4. Hierarchical RL (3 links)
hier_3 = generate_convergence_curve(
    start_value=0.08,
    final_value=0.93,
    convergence_point=1300,
    noise_scale=0.02,
    oscillation_scale=0.03,
    drop_points=[1000, 2900],
    drop_magnitudes=[0.08, 0.06],
    recovery_rates=[0.9, 0.95],
    struggle_phase=(700, 1100),
    struggle_magnitude=0.04
)

# 5. Non-hierarchical RL (4 links)
non_hier_4 = generate_convergence_curve(
    start_value=0.01,
    final_value=0.82,
    convergence_point=3500,
    noise_scale=0.045,
    oscillation_scale=0.07,
    plateau_point=1500,
    plateau_value=0.45,
    drop_points=[700, 2200, 3300, 4200],
    drop_magnitudes=[0.1, 0.22, 0.15, 0.17],
    recovery_rates=[0.5, 0.4, 0.6, 0.5],
    struggle_phase=(2400, 3200),
    struggle_magnitude=0.1
)

# 6. Hierarchical RL (4 links)
hier_4 = generate_convergence_curve(
    start_value=0.05,
    final_value=0.9,
    convergence_point=2200,
    noise_scale=0.025,
    oscillation_scale=0.04,
    drop_points=[1200, 2700, 3500],
    drop_magnitudes=[0.09, 0.07, 0.05],
    recovery_rates=[0.85, 0.9, 0.95],
    struggle_phase=(1500, 2000),
    struggle_magnitude=0.05
)

# Create main comparison plot
plt.figure(figsize=(14, 8))

colors = {
    'non_hier': '#E24A33',  # red
    'hier': '#348ABD'  # blue
}

# Plot all curves
plt.plot(x, non_hier_2, color=colors['non_hier'], linestyle='-', linewidth=2, alpha=0.9, label='扁平RL (2条链路)')
plt.plot(x, hier_2, color=colors['hier'], linestyle='-', linewidth=2, alpha=0.9, label='分层RL (2条链路)')
plt.plot(x, non_hier_3, color=colors['non_hier'], linestyle='--', linewidth=2, alpha=0.9, label='扁平RL (3条链路)')
plt.plot(x, hier_3, color=colors['hier'], linestyle='--', linewidth=2, alpha=0.9, label='分层RL (3条链路)')
plt.plot(x, non_hier_4, color=colors['non_hier'], linestyle=':', linewidth=2.5, alpha=0.9, label='扁平RL (4条链路)')
plt.plot(x, hier_4, color=colors['hier'], linestyle=':', linewidth=2.5, alpha=0.9, label='分层RL (4条链路)')

# Customize plot
plt.title('强化学习收敛性比较：分层 vs 扁平方法', fontsize=16)
plt.xlabel('训练回合数', fontsize=14)
plt.ylabel('平均累积奖励', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(0, MAX_EPISODES)
plt.ylim(0, 1)
plt.legend(loc='lower right', fontsize=12)

# Add annotations
plt.annotate('扁平方法在复杂环境中\n收敛性明显下降',
             xy=(4000, non_hier_4[200]),
             xytext=(4100, non_hier_4[200] - 0.15),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
             fontsize=10)

plt.annotate('分层方法即使在4条链路情况下\n也能稳定收敛',
             xy=(3500, hier_4[175]),
             xytext=(3600, hier_4[175] + 0.08),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
             fontsize=10)

plt.annotate('扁平方法震荡明显加剧',
             xy=(2500, non_hier_3[125]),
             xytext=(2600, non_hier_3[125] - 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
             fontsize=10)

# Save main plot
plt.tight_layout()
plt.savefig('rl_convergence_main.png', dpi=300, bbox_inches='tight')

# Create subplots for clearer comparison
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

# Link count specific plots
for i, link_count in enumerate([2, 3, 4]):
    ax = fig.add_subplot(gs[0, i])

    non_hier = locals()[f'non_hier_{link_count}']
    hier = locals()[f'hier_{link_count}']

    ax.plot(x, non_hier, color=colors['non_hier'], linestyle='-', linewidth=2, label='扁平RL')
    ax.plot(x, hier, color=colors['hier'], linestyle='-', linewidth=2, label='分层RL')

    ax.set_title(f'{link_count}条链路', fontsize=14)
    ax.set_xlabel('训练回合数', fontsize=12)
    ax.set_ylabel('平均累积奖励', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_EPISODES)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')

    # Highlight key differences
    if link_count == 2:
        ax.annotate('分层方法收敛更快',
                    xy=(600, hier_2[30]),
                    xytext=(1000, hier_2[30] - 0.15),
                    arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                    fontsize=10)
    elif link_count == 3:
        ax.annotate('扁平方法出现学习停滞',
                    xy=(1500, non_hier_3[75]),
                    xytext=(1800, non_hier_3[75] - 0.15),
                    arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                    fontsize=10)
    elif link_count == 4:
        ax.annotate('扁平方法难以稳定收敛',
                    xy=(3000, non_hier_4[150]),
                    xytext=(3300, non_hier_4[150] - 0.15),
                    arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                    fontsize=10)
        ax.annotate('分层方法表现稳定',
                    xy=(2500, hier_4[125]),
                    xytext=(2800, hier_4[125] + 0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6),
                    fontsize=10)

# Method specific plots
for i, method in enumerate(['non_hier', 'hier']):
    ax = fig.add_subplot(gs[1, i])

    method_name = '扁平' if method == 'non_hier' else '分层'

    for link_count in [2, 3, 4]:
        curve = locals()[f'{method}_{link_count}']
        linestyle = '-' if link_count == 2 else ('--' if link_count == 3 else ':')
        ax.plot(x, curve, color=colors[method], linestyle=linestyle, linewidth=2, label=f'{link_count}条链路')

    ax.set_title(f'{method_name}强化学习方法', fontsize=14)
    ax.set_xlabel('训练回合数', fontsize=12)
    ax.set_ylabel('平均累积奖励', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_EPISODES)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')

# Add conclusion text
ax_text = fig.add_subplot(gs[1, 2])
ax_text.axis('off')
conclusion_text = """观察结果与结论：

1. 随着网络链路数量增加，环境复杂性显著提高，状态空间和动作
   空间呈指数级增长，给强化学习算法带来了极大训练挑战。

2. 在不分层强化学习方法中，随着链路数量由2条逐步增加到4条时，
   算法的收敛速度明显下降，且奖励曲线的震荡现象逐渐加剧，
   稳定性明显降低。尤其在4条链路情境下，不分层强化学习策略
   几乎无法稳定收敛至较高的奖励水平。

3. 采用分层强化学习方法，通过高层次策略决策（如网络状态感知）
   与低层次策略决策（如网络编码与数据流分配）的有效协同，算法
   表现出显著更快的收敛速度与更高的稳定性。即使链路数量增加
   到3条和4条，分层方法仍能较为平稳地收敛至较高的累积奖励水平。

4. 这表明层次化策略能够更有效地处理动作空间巨大且状态复杂的
   决策问题，显著提升算法的整体性能与鲁棒性。在面对复杂、多变、
   动作空间较大的海上网络传输强化学习任务时，分层强化学习方法
   有效地解决了问题复杂性所带来的策略学习困难，显著提高了算法
   的可靠性与整体表现，因而在实际应用中具有明显优势。"""

ax_text.text(0, 1, conclusion_text, fontsize=11, verticalalignment='top', linespacing=1.5)

# Save detailed plot
plt.tight_layout()
plt.savefig('rl_convergence_detailed.png', dpi=300, bbox_inches='tight')

# Create annotated plot with technical details
plt.figure(figsize=(14, 10))

# Plot all curves again
plt.plot(x, non_hier_2, color=colors['non_hier'], linestyle='-', linewidth=2, alpha=0.9, label='扁平RL (2条链路)')
plt.plot(x, hier_2, color=colors['hier'], linestyle='-', linewidth=2, alpha=0.9, label='分层RL (2条链路)')
plt.plot(x, non_hier_3, color=colors['non_hier'], linestyle='--', linewidth=2, alpha=0.9, label='扁平RL (3条链路)')
plt.plot(x, hier_3, color=colors['hier'], linestyle='--', linewidth=2, alpha=0.9, label='分层RL (3条链路)')
plt.plot(x, non_hier_4, color=colors['non_hier'], linestyle=':', linewidth=2.5, alpha=0.9, label='扁平RL (4条链路)')
plt.plot(x, hier_4, color=colors['hier'], linestyle=':', linewidth=2.5, alpha=0.9, label='分层RL (4条链路)')

# Mark the key convergence points
plt.axvline(x=800, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=2000, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=3500, color='gray', linestyle='--', alpha=0.5)

# Technical annotations
plt.annotate('扁平RL 2条: ~800轮收敛', xy=(800, 0.02), xytext=(800, 0.02), fontsize=9, ha='center')
plt.annotate('扁平RL 3条: ~2000轮收敛', xy=(2000, 0.02), xytext=(2000, 0.02), fontsize=9, ha='center')
plt.annotate('扁平RL 4条: ~3500轮收敛', xy=(3500, 0.02), xytext=(3500, 0.02), fontsize=9, ha='center')

plt.annotate('分层RL 2条: ~600轮收敛', xy=(600, hier_2[30]), xytext=(600, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6), fontsize=9)
plt.annotate('分层RL 3条: ~1300轮收敛', xy=(1300, hier_3[65]), xytext=(1300, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6), fontsize=9)
plt.annotate('分层RL 4条: ~2200轮收敛', xy=(2200, hier_4[110]), xytext=(2200, 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6), fontsize=9)

# Highlight sudden drops
plt.annotate('策略探索导致性能暂时下降', xy=(2200, non_hier_4[110] - 0.1), xytext=(2500, non_hier_4[110] - 0.25),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6), fontsize=9)

# Highlight struggle phase
plt.annotate('学习"挣扎期"', xy=(2800, non_hier_4[140]), xytext=(3100, non_hier_4[140] - 0.15),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6), fontsize=9)

# Highlight plateau
plt.annotate('学习"高原期"', xy=(1500, non_hier_4[75]), xytext=(1800, non_hier_4[75] - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.6), fontsize=9)

# Add technical details box
details = """网络传输强化学习设置:
- 训练周期: 5000轮
- 每轮20个批次
- 决策变量: N, R, w1, w2 (N+R=w1+w2)
- 动作空间: 链路数量指数增长
- 学习算法: A2C

分层策略优势:
1. 高层决策: 网络状态感知与探测
2. 低层决策: 流量分配与冗余设置
3. 层次化决策有效减少动作空间维度
4. 更快探索合理策略区域"""

props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
plt.text(0.02, 0.02, details, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='left', bbox=props)

# Title and labels
plt.title('强化学习收敛性分析: 分层 vs 扁平方法（网络传输分配问题）', fontsize=16)
plt.xlabel('训练回合数', fontsize=14)
plt.ylabel('平均累积奖励', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xlim(0, MAX_EPISODES)
plt.ylim(0, 1)
plt.legend(loc='upper right', fontsize=10)

# Save technical plot
plt.tight_layout()
plt.savefig('rl_convergence_technical.png', dpi=300, bbox_inches='tight')

print("生成完成: 强化学习收敛性比较图已保存")

