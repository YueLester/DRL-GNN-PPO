import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle, Polygon
import math
import random
from scipy.spatial.distance import cdist
import time

# 设置随机种子确保可重现
np.random.seed(42)
random.seed(42)


class GWOSolver:
    """
    基于灰狼算法(GWO)的机会无人机接入网络上传节点选择求解器
    """

    def __init__(self, params=None):
        """
        初始化求解器参数
        """
        if params is None:
            params = {}

        # 算法参数
        self.population_size = params.get('population_size', 30)  # 灰狼种群大小
        self.max_iterations = params.get('max_iterations', 100)  # 最大迭代次数
        self.dimensions = params.get('dimensions', 50)  # 节点数(问题维度)
        self.a_decrease_coef = params.get('a_decrease_coef', 2)  # a参数递减系数

        # 问题参数
        self.node_positions = params.get('node_positions', self._generate_random_positions(self.dimensions))
        self.uav_trajectory = params.get('uav_trajectory', {'start': (0, 500), 'end': (1000, 500)})
        self.data_rates = params.get('data_rates', np.random.uniform(1, 10, self.dimensions))

        # 适应度函数权重
        self.weights = {
            'upload_rate': params.get('w_upload_rate', 0.4),  # 无人机接入性能权重(a1)
            'node_density': params.get('w_node_density', 0.3),  # 节点分布与密度权重(μ)
            'aggregation_cost': params.get('w_agg_cost', 0.2),  # 数据聚合代价权重(β)
            'cluster_head_count': params.get('w_ch_count', 0.1)  # 簇头数量控制权重(γ)
        }

        # 初始化种群相关变量
        self.wolves = []  # 灰狼种群
        self.alpha_wolf = None  # α狼(最优解)
        self.beta_wolf = None  # β狼(次优解)
        self.delta_wolf = None  # δ狼(第三优解)

        # 优化结果记录
        self.convergence_curve = []  # 收敛曲线
        self.best_solution = None  # 最优解
        self.best_fitness = -np.inf  # 最优适应度值

        # 预计算性能优化
        self.distance_matrix = self._compute_distance_matrix()
        self.uav_distances = self._compute_uav_distances()

    def _generate_random_positions(self, n):
        """
        随机生成节点位置
        """
        return np.random.uniform(0, 1000, (n, 2))

    def _compute_distance_matrix(self):
        """
        计算节点间距离矩阵(性能优化)
        """
        return cdist(self.node_positions, self.node_positions)

    def _compute_uav_distances(self):
        """
        计算节点到UAV轨迹的距离
        """
        distances = np.zeros(self.dimensions)
        start = np.array(self.uav_trajectory['start'])
        end = np.array(self.uav_trajectory['end'])

        # 计算轨迹方向向量
        trajectory_vec = end - start
        trajectory_len_squared = np.sum(trajectory_vec ** 2)

        for i in range(self.dimensions):
            node_pos = self.node_positions[i]

            # 处理轨迹为单点的情况
            if trajectory_len_squared == 0:
                distances[i] = np.linalg.norm(node_pos - start)
                continue

            # 计算节点在轨迹上的投影点
            t = max(0, min(1, np.dot(node_pos - start, trajectory_vec) / trajectory_len_squared))
            projection = start + t * trajectory_vec

            # 节点到投影点的距离
            distances[i] = np.linalg.norm(node_pos - projection)

        return distances

    def calculate_fitness(self, solution):
        """
        计算个体适应度
        """
        # 将解向量转换为布尔数组以方便索引
        solution_binary = np.array(solution, dtype=bool)

        # 获取所有簇头索引
        cluster_heads = np.where(solution_binary)[0]

        if len(cluster_heads) == 0:
            return -np.inf  # 无簇头，返回极低适应度

        # 1. 计算上传率 (F_rate)
        distances_to_uav = self.uav_distances[cluster_heads]
        comm_quality = np.maximum(0, 1 - distances_to_uav / 300)
        upload_rate = np.sum(self.data_rates[cluster_heads] * comm_quality)

        # 2. 计算节点密度 (ρ)
        node_density = 0
        for head_idx in cluster_heads:
            # 计算该簇头到所有其他节点的平均距离倒数
            distances = self.distance_matrix[head_idx]
            avg_dist = np.mean(np.delete(distances, head_idx))
            node_density += 1 / (avg_dist + 1e-10)  # 避免除以0

        # 3. 计算汇聚代价 (C_agg)
        aggregation_cost = 0
        non_ch_indices = np.where(~solution_binary)[0]

        if len(non_ch_indices) > 0:
            # 为每个非簇头节点找到最近的簇头
            for node_idx in non_ch_indices:
                min_dist = np.min(self.distance_matrix[node_idx, cluster_heads])
                # 根据到UAV距离计算权重
                weight = 1 / (1 + 0.01 * self.uav_distances[node_idx])
                aggregation_cost += weight * min_dist

        # 4. 簇头数量 (|H_n|)
        cluster_head_count = len(cluster_heads)

        # 计算总适应度: fitness = a1·F_rate + μ·∑ρ(k) - β·∑w_i·P - γ·|H_n|
        fitness = (
                self.weights['upload_rate'] * upload_rate +
                self.weights['node_density'] * node_density -
                self.weights['aggregation_cost'] * aggregation_cost -
                self.weights['cluster_head_count'] * cluster_head_count
        )

        return fitness

    def initialize_population(self):
        """
        初始化灰狼种群(启发式初始化)
        """
        self.wolves = []

        for _ in range(self.population_size):
            # 距离UAV越近的节点更可能成为簇头
            probabilities = np.exp(-self.uav_distances / 200)
            solution = (np.random.random(self.dimensions) < probabilities).astype(int)

            # 确保至少有一个簇头
            if np.sum(solution) == 0:
                random_idx = np.random.randint(0, self.dimensions)
                solution[random_idx] = 1

            fitness = self.calculate_fitness(solution)
            self.wolves.append({
                'position': solution.copy(),
                'fitness': fitness
            })

        # 更新领导狼
        self._update_leader_wolves()

    def _update_leader_wolves(self):
        """
        更新α、β、δ三只领导狼
        """
        # 按适应度降序排序
        self.wolves.sort(key=lambda x: x['fitness'], reverse=True)

        # 更新领导者
        self.alpha_wolf = self.wolves[0].copy()
        self.beta_wolf = self.wolves[1].copy()
        self.delta_wolf = self.wolves[2].copy()

        # 更新全局最优解
        if self.alpha_wolf['fitness'] > self.best_fitness:
            self.best_fitness = self.alpha_wolf['fitness']
            self.best_solution = self.alpha_wolf['position'].copy()

    def _update_position(self, current_position, a):
        """
        二进制灰狼位置更新
        """
        new_position = np.zeros(self.dimensions, dtype=int)

        for j in range(self.dimensions):
            # 基于α、β、δ狼计算新位置
            r1 = np.random.random()
            r2 = np.random.random()

            d_alpha = abs(2 * r1 * self.alpha_wolf['position'][j] - current_position[j])
            d_beta = abs(2 * r1 * self.beta_wolf['position'][j] - current_position[j])
            d_delta = abs(2 * r1 * self.delta_wolf['position'][j] - current_position[j])

            x1 = self.alpha_wolf['position'][j] - a * d_alpha
            x2 = self.beta_wolf['position'][j] - a * d_beta
            x3 = self.delta_wolf['position'][j] - a * d_delta

            # 取三个狼位置的平均影响
            x_avg = (x1 + x2 + x3) / 3

            # 二进制GWO: 使用S型转移函数将连续值转换为二进制
            s = 1 / (1 + np.exp(-10 * (x_avg - 0.5)))

            # 基于概率更新位置
            new_position[j] = 1 if np.random.random() < s else 0

        # 确保至少有一个簇头
        if np.sum(new_position) == 0:
            random_idx = np.random.randint(0, self.dimensions)
            new_position[random_idx] = 1

        return new_position

    def optimize(self):
        """
        运行灰狼优化算法
        """
        print("开始优化...")
        start_time = time.time()

        # 初始化种群
        self.initialize_population()

        # 迭代优化
        for iteration in range(self.max_iterations):
            # 更新a参数(从2线性递减到0)
            a = self.a_decrease_coef * (1 - iteration / self.max_iterations)

            # 更新每只狼的位置
            for i in range(3, self.population_size):  # 跳过α、β、δ狼
                new_position = self._update_position(self.wolves[i]['position'], a)
                new_fitness = self.calculate_fitness(new_position)

                self.wolves[i]['position'] = new_position
                self.wolves[i]['fitness'] = new_fitness

            # 更新领导狼
            self._update_leader_wolves()

            # 记录收敛曲线数据
            self.convergence_curve.append(self.best_fitness)

            # 输出迭代状态
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                ch_count = np.sum(self.best_solution)
                print(f"迭代 {iteration + 1}/{self.max_iterations}: 最佳适应度 = {self.best_fitness:.4f}, 簇头数量 = {ch_count}")

        # 计算最终聚类结果
        self.clusters = self._get_clustering_result()

        end_time = time.time()
        print(f"优化完成! 用时: {end_time - start_time:.2f} 秒")

        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'convergence_curve': self.convergence_curve,
            'clusters': self.clusters
        }

    def _get_clustering_result(self):
        """
        获取最终的聚类结果
        """
        clusters = []
        solution_binary = np.array(self.best_solution, dtype=bool)

        # 获取所有簇头索引
        cluster_heads = np.where(solution_binary)[0]

        # 初始化簇
        for i, head_idx in enumerate(cluster_heads):
            clusters.append({
                'id': i,
                'head': int(head_idx),
                'members': [int(head_idx)],
                'position': self.node_positions[head_idx]
            })

        # 为每个非簇头节点分配最近的簇头
        non_ch_indices = np.where(~solution_binary)[0]

        for node_idx in non_ch_indices:
            distances = self.distance_matrix[node_idx, cluster_heads]
            closest_ch_idx = np.argmin(distances)
            clusters[closest_ch_idx]['members'].append(int(node_idx))

        return clusters

    def plot_convergence(self, save_path=None):
        """
        绘制收敛曲线
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.convergence_curve) + 1), self.convergence_curve, 'b-', linewidth=2)
        plt.title('GWO算法收敛曲线', fontsize=15)
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('适应度值', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"收敛曲线已保存至: {save_path}")

        plt.show()

    def plot_clustering(self, save_path=None):
        """
        绘制聚类组网示意图
        """
        plt.figure(figsize=(12, 8))

        # 绘制UAV轨迹
        uav_start = self.uav_trajectory['start']
        uav_end = self.uav_trajectory['end']
        plt.plot([uav_start[0], uav_end[0]], [uav_start[1], uav_end[1]],
                 'k--', linewidth=2, label='UAV轨迹')

        # 添加UAV图标
        uav_icon = plt.Circle((500, 500), 20, color='black', alpha=0.5)
        plt.gca().add_patch(uav_icon)
        plt.text(500, 530, 'UAV', ha='center', fontsize=10)

        # 绘制不同簇的节点和连接线
        colors = cm.rainbow(np.linspace(0, 1, len(self.clusters)))

        for i, cluster in enumerate(self.clusters):
            head_idx = cluster['head']
            head_pos = self.node_positions[head_idx]
            members = cluster['members']

            # 绘制簇头
            plt.scatter(head_pos[0], head_pos[1], s=100, c=[colors[i]],
                        marker='*', edgecolors='k', linewidths=1,
                        label=f'簇头 {head_idx}')

            # 围绕簇头添加圆圈
            circle = Circle((head_pos[0], head_pos[1]), 20, fill=False,
                            linestyle='-', linewidth=1.5, color=colors[i])
            plt.gca().add_patch(circle)

            # 绘制簇成员和连接线
            for member_idx in members:
                if member_idx != head_idx:  # 跳过簇头自身
                    member_pos = self.node_positions[member_idx]

                    # 绘制簇成员节点
                    plt.scatter(member_pos[0], member_pos[1], s=50, c=[colors[i]],
                                alpha=0.6, edgecolors='k', linewidths=0.5)

                    # 绘制成员到簇头的连接线
                    plt.plot([member_pos[0], head_pos[0]], [member_pos[1], head_pos[1]],
                             '-', color=colors[i], linewidth=0.8, alpha=0.5)

            # 绘制簇头到UAV轨迹的连接线(上传链路)
            # 计算簇头在UAV轨迹上的投影点
            trajectory_vec = np.array(uav_end) - np.array(uav_start)
            t = max(0, min(1, np.dot(head_pos - np.array(uav_start), trajectory_vec) / np.sum(trajectory_vec ** 2)))
            projection = np.array(uav_start) + t * trajectory_vec

            plt.plot([head_pos[0], projection[0]], [head_pos[1], projection[1]],
                     '--', color=colors[i], linewidth=2)

        # 设置图表属性
        plt.title('机会无人机接入网络聚类示意图', fontsize=15)
        plt.xlabel('X坐标', fontsize=12)
        plt.ylabel('Y坐标', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)

        # 添加图例
        plt.legend(loc='upper right', fontsize=10)

        # 设置坐标轴范围
        plt.xlim(-50, 1050)
        plt.ylim(-50, 1050)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"聚类组网示意图已保存至: {save_path}")

        plt.show()


# 主函数
def main():
    # 设置问题参数
    params = {
        'population_size': 30,  # 灰狼种群大小
        'max_iterations': 1000,  # 最大迭代次数
        'dimensions': 200,  # 网络节点数量
        'w_upload_rate': 0.4,  # 上传性能权重
        'w_node_density': 0.3,  # 节点密度权重
        'w_agg_cost': 0.2,  # 汇聚代价权重
        'w_ch_count': 0.1  # 簇头数量控制权重
    }

    # 创建求解器实例
    solver = GWOSolver(params)

    # 运行优化
    result = solver.optimize()

    # 打印最佳解信息
    print("\n优化结果:")
    print(f"最佳适应度: {result['best_fitness']:.4f}")
    print(f"簇头数量: {np.sum(result['best_solution'])}")
    print(f"簇的数量: {len(result['clusters'])}")

    # 绘制收敛曲线和聚类示意图
    solver.plot_convergence(save_path="gwo_convergence.png")
    solver.plot_clustering(save_path="gwo_clustering.png")


if __name__ == "__main__":
    main()