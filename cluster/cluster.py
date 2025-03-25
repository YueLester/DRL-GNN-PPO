import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from scipy.spatial import distance

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# 参数设置
num_nodes = 100  # 网络节点数
num_clusters = 30  # 簇的数量
network_size = 1000  # 网络区域大小
num_drones = 2  # 无人机数量
num_base_stations = 3  # 基站数量
num_iterations = 100  # 算法迭代次数
population_size = 30  # 粒子群大小


# 节点属性
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = random.uniform(0.5, 1.0)  # 初始能量随机分配
        self.degree = 0  # 节点度
        self.cluster = -1  # 所属簇


# 无人机轨迹
class Drone:
    def __init__(self, trajectory):
        self.trajectory = trajectory  # 轨迹点列表
        self.current_pos = 0  # 当前位置索引

    def move(self):
        self.current_pos = (self.current_pos + 1) % len(self.trajectory)
        return self.trajectory[self.current_pos]

    def get_position(self):
        return self.trajectory[self.current_pos]


# 基站
class BaseStation:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# 创建网络节点
nodes = []
for i in range(num_nodes):
    x = random.uniform(0, network_size)
    y = random.uniform(0, network_size)
    nodes.append(Node(x, y))

# 创建基站
base_stations = []
for i in range(num_base_stations):
    x = random.uniform(0, network_size)
    y = random.uniform(0, network_size)
    base_stations.append(BaseStation(x, y))

# 创建无人机轨迹
drones = []
for i in range(num_drones):
    trajectory_points = []
    # 生成圆形轨迹
    center_x = random.uniform(200, network_size - 200)
    center_y = random.uniform(200, network_size - 200)
    radius = random.uniform(100, 200)
    num_points = 20  # 轨迹点数量

    for j in range(num_points):
        angle = 2 * np.pi * j / num_points
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        trajectory_points.append((x, y))

    drones.append(Drone(trajectory_points))


# 计算节点度
def calculate_node_degree(nodes, comm_range=100):
    for i in range(len(nodes)):
        degree = 0
        for j in range(len(nodes)):
            if i != j:
                dist = np.sqrt((nodes[i].x - nodes[j].x) ** 2 + (nodes[i].y - nodes[j].y) ** 2)
                if dist <= comm_range:
                    degree += 1
        nodes[i].degree = degree


calculate_node_degree(nodes)


# 粒子群优化算法
class PSO:
    def __init__(self, nodes, drones, base_stations):
        self.nodes = nodes
        self.drones = drones
        self.base_stations = base_stations
        self.particles = []
        self.gbest = None
        self.gbest_fitness = float('-inf')
        self.fitness_history = []

        # 初始化粒子
        for _ in range(population_size):
            particle = np.random.randint(0, num_clusters, num_nodes)
            velocity = np.zeros(num_nodes)
            pbest = particle.copy()
            pbest_fitness = self.calculate_fitness(particle)

            self.particles.append({
                'position': particle,
                'velocity': velocity,
                'pbest': pbest,
                'pbest_fitness': pbest_fitness
            })

            if pbest_fitness > self.gbest_fitness:
                self.gbest = pbest.copy()
                self.gbest_fitness = pbest_fitness

    def calculate_fitness(self, particle):
        # 将粒子映射到节点的簇分配
        for i in range(num_nodes):
            self.nodes[i].cluster = particle[i]

        # 选择簇头
        cluster_heads = [-1] * num_clusters
        for c in range(num_clusters):
            cluster_nodes = [i for i in range(num_nodes) if self.nodes[i].cluster == c]
            if not cluster_nodes:
                continue

            # 根据能量、节点度、到无人机和基站的距离选择簇头
            best_score = float('-inf')
            for node_idx in cluster_nodes:
                node = self.nodes[node_idx]

                # 计算到无人机的距离
                min_drone_dist = float('inf')
                for drone in self.drones:
                    drone_pos = drone.get_position()
                    dist = np.sqrt((node.x - drone_pos[0]) ** 2 + (node.y - drone_pos[1]) ** 2)
                    min_drone_dist = min(min_drone_dist, dist)

                # 计算到基站的距离
                min_bs_dist = float('inf')
                for bs in self.base_stations:
                    dist = np.sqrt((node.x - bs.x) ** 2 + (node.y - bs.y) ** 2)
                    min_bs_dist = min(min_bs_dist, dist)

                # 综合考虑各因素的得分
                score = (node.energy * 0.4 +
                         node.degree * 0.2 +
                         (1 / (min_drone_dist + 1)) * 0.2 +
                         (1 / (min_bs_dist + 1)) * 0.2)

                if score > best_score:
                    best_score = score
                    cluster_heads[c] = node_idx

        # 计算簇内凝聚度和簇间分离度的适应度
        intra_cluster_distance = 0
        inter_cluster_distance = 0

        # 计算簇内距离
        for i in range(num_nodes):
            cluster = self.nodes[i].cluster
            if cluster_heads[cluster] != -1:
                head = self.nodes[cluster_heads[cluster]]
                intra_cluster_distance += np.sqrt((self.nodes[i].x - head.x) ** 2 +
                                                  (self.nodes[i].y - head.y) ** 2)

        # 计算簇间距离
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                if cluster_heads[i] != -1 and cluster_heads[j] != -1:
                    head_i = self.nodes[cluster_heads[i]]
                    head_j = self.nodes[cluster_heads[j]]
                    inter_cluster_distance += np.sqrt((head_i.x - head_j.x) ** 2 +
                                                      (head_i.y - head_j.y) ** 2)

        # 计算适应度
        if intra_cluster_distance == 0:
            return 0

        return inter_cluster_distance / (intra_cluster_distance + 1)

    def update(self):
        w = 0.7  # 惯性权重
        c1 = 1.4  # 个体学习因子
        c2 = 1.4  # 社会学习因子

        for particle in self.particles:
            # 更新速度
            r1 = np.random.random(num_nodes)
            r2 = np.random.random(num_nodes)

            cognitive_velocity = c1 * r1 * (particle['pbest'] - particle['position'])
            social_velocity = c2 * r2 * (self.gbest - particle['position'])

            particle['velocity'] = w * particle['velocity'] + cognitive_velocity + social_velocity

            # 更新位置
            particle['position'] = particle['position'] + particle['velocity']
            particle['position'] = np.round(particle['position']).astype(int)
            particle['position'] = np.clip(particle['position'], 0, num_clusters - 1)

            # 更新个体最优
            fitness = self.calculate_fitness(particle['position'])
            if fitness > particle['pbest_fitness']:
                particle['pbest'] = particle['position'].copy()
                particle['pbest_fitness'] = fitness

                # 更新全局最优
                if fitness > self.gbest_fitness:
                    self.gbest = particle['position'].copy()
                    self.gbest_fitness = fitness

        self.fitness_history.append(self.gbest_fitness)

    def run(self, iterations):
        for _ in range(iterations):
            self.update()

        # 应用最终的簇分配结果
        for i in range(num_nodes):
            self.nodes[i].cluster = self.gbest[i]


# 运行粒子群优化算法
pso = PSO(nodes, drones, base_stations)
pso.run(num_iterations)



# 绘制结果
plt.figure(figsize=(15, 10))

# 绘制分簇结果
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown']
markers = ['o', 'v', 'P', '*', 'X', 'd', 'p']

# 绘制簇及其成员
for i in range(num_nodes):
    plt.scatter(nodes[i].x, nodes[i].y,
                c=colors[nodes[i].cluster % len(colors)],
                marker=markers[nodes[i].cluster % len(markers)],
                alpha=0.7,
                s=50)

# 找出并标记簇头
cluster_heads = {}
for c in range(num_clusters):
    cluster_nodes = [i for i in range(num_nodes) if nodes[i].cluster == c]
    if not cluster_nodes:
        continue

    best_score = float('-inf')
    best_head = -1
    for node_idx in cluster_nodes:
        node = nodes[node_idx]

        # 计算到无人机的距离
        min_drone_dist = float('inf')
        for drone in drones:
            drone_pos = drone.get_position()
            dist = np.sqrt((node.x - drone_pos[0]) ** 2 + (node.y - drone_pos[1]) ** 2)
            min_drone_dist = min(min_drone_dist, dist)

        # 计算到基站的距离
        min_bs_dist = float('inf')
        for bs in base_stations:
            dist = np.sqrt((node.x - bs.x) ** 2 + (node.y - bs.y) ** 2)
            min_bs_dist = min(min_bs_dist, dist)

        # 综合考虑各因素的得分
        score = (node.energy * 0.4 +
                 node.degree * 0.2 +
                 (1 / (min_drone_dist + 1)) * 0.2 +
                 (1 / (min_bs_dist + 1)) * 0.2)

        if score > best_score:
            best_score = score
            best_head = node_idx

    if best_head != -1:
        cluster_heads[c] = best_head

# 绘制簇头
for c, head_idx in cluster_heads.items():
    plt.scatter(nodes[head_idx].x, nodes[head_idx].y,
                c=colors[c % len(colors)],
                marker='h',
                s=200,
                edgecolor='black',
                linewidth=2,
                label=f'簇{c + 1}簇头')

    # 绘制从簇成员到簇头的连线
    for i in range(num_nodes):
        if nodes[i].cluster == c and i != head_idx:
            plt.plot([nodes[i].x, nodes[head_idx].x],
                     [nodes[i].y, nodes[head_idx].y],
                     c=colors[c % len(colors)],
                     linestyle='-',
                     alpha=0.2)

# 绘制基站
for i, bs in enumerate(base_stations):
    plt.scatter(bs.x, bs.y, c='black', marker='^', s=200, label=f'基站{i + 1}')

    # 绘制从簇头到最近基站的连线
    for c, head_idx in cluster_heads.items():
        head = nodes[head_idx]
        plt.plot([head.x, bs.x], [head.y, bs.y], 'k-.', alpha=0.5)

# 绘制无人机轨迹
for i, drone in enumerate(drones):
    traj_x = [point[0] for point in drone.trajectory]
    traj_y = [point[1] for point in drone.trajectory]
    plt.plot(traj_x, traj_y, 'k--')
    plt.scatter(traj_x, traj_y, c='yellow', alpha=0.3)
    current_pos = drone.get_position()
    plt.scatter(current_pos[0], current_pos[1], c='yellow', marker='s', s=100, label=f'无人机{i + 1}')

plt.title('网络分簇结果', fontsize=16)
plt.xlabel('X坐标', fontsize=14)
plt.ylabel('Y坐标', fontsize=14)
plt.grid(True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
plt.savefig('cluster_result.png', dpi=300, bbox_inches='tight')

# 添加文本标注
plt.figtext(0.02, 0.02, f"总节点数: {num_nodes}, 簇数量: {num_clusters}, 无人机数量: {num_drones}, 基站数量: {num_base_stations}",
            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pso.fitness_history) + 1), pso.fitness_history, 'b-', linewidth=2)
plt.title('PSO算法收敛曲线', fontsize=16)
plt.xlabel('迭代次数', fontsize=14)
plt.ylabel('适应度值', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('convergence.png', dpi=300, bbox_inches='tight')

plt.show()
