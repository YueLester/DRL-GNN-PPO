import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D
import math
import random
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import time

# 设置随机种子确保可重现
np.random.seed(42)
random.seed(42)


class NetworkOptimizer:
    """
    无人机接入网络组网算法比较器
    """

    def __init__(self, params=None):
        if params is None:
            params = {}

        # 网络参数
        self.num_nodes = params.get('num_nodes', 40)
        self.area_size = params.get('area_size', 1000)  # 区域大小 (m)
        self.uav_trajectory = params.get('uav_trajectory', {
            'start': (500, -200),  # UAV起始位置
            'end': (500, 1200)  # UAV结束位置
        })

        # 生成节点
        self.generate_nodes()

        # 计算距离矩阵
        self.distance_matrix = self.compute_distance_matrix()
        self.distances_to_uav = self.compute_uav_distances()

        # 聚类结果
        self.proposed_clusters = None
        self.kmeans_clusters = None
        self.self_organized_clusters = None

    def generate_nodes(self):
        """
        生成具有聚类特性的节点
        """
        # 创建四个聚类区域
        cluster_centers = [
            (200, 800),  # Co1 - 左上
            (200, 200),  # Co2 - 左下
            (800, 800),  # Co3 - 右上
            (800, 200)  # Co4 - 右下
        ]

        # 为每个聚类生成节点
        nodes = []
        node_ids = []
        node_types = []  # 1=普通节点, 2=潜在簇头
        cluster_labels = []

        nodes_per_cluster = self.num_nodes // 4

        for i, center in enumerate(cluster_centers):
            # 在聚类中心周围生成节点
            cluster_nodes = np.random.normal(center, scale=100, size=(nodes_per_cluster, 2))

            # 给每个节点分配ID (u1, u2, ...)
            for j in range(nodes_per_cluster):
                node_id = f"u{len(nodes) + 1}"
                node_ids.append(node_id)

                # 确定节点类型 (有约20%的机会成为潜在簇头)
                if np.random.random() < 0.2:
                    node_types.append(2)  # 潜在簇头
                else:
                    node_types.append(1)  # 普通节点

                cluster_labels.append(i)  # 记录节点所属的聚类

            nodes.extend(cluster_nodes)

        # 转换为numpy数组
        self.node_positions = np.array(nodes)
        self.node_ids = np.array(node_ids)
        self.node_types = np.array(node_types)
        self.cluster_labels = np.array(cluster_labels)

    def compute_distance_matrix(self):
        """
        计算节点间距离矩阵
        """
        return cdist(self.node_positions, self.node_positions)

    def compute_uav_distances(self):
        """
        计算节点到UAV轨迹的距离
        """
        distances = np.zeros(self.num_nodes)
        start = np.array(self.uav_trajectory['start'])
        end = np.array(self.uav_trajectory['end'])

        # 计算轨迹方向向量
        trajectory_vec = end - start
        trajectory_len_squared = np.sum(trajectory_vec ** 2)

        for i in range(self.num_nodes):
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

    def proposed_algorithm(self):
        """
        实现灰狼优化的无人机网络接入算法(所提组网策略)
        """
        print("运行所提组网策略...")

        # 灰狼优化算法参数
        population_size = 30
        max_iterations = 50

        # 初始化种群
        wolves = []
        for _ in range(population_size):
            # 距离UAV越近的节点更可能成为簇头
            probabilities = np.exp(-self.distances_to_uav / 200)
            probabilities[self.node_types == 1] *= 0.5  # 降低普通节点成为簇头的概率

            solution = (np.random.random(self.num_nodes) < probabilities).astype(int)

            # 确保至少有四个簇头(每个区域一个)
            for cluster_id in range(4):
                cluster_nodes = np.where(self.cluster_labels == cluster_id)[0]
                if np.sum(solution[cluster_nodes]) == 0:
                    # 在该区域随机选择一个节点作为簇头
                    potential_heads = cluster_nodes[self.node_types[cluster_nodes] == 2]
                    if len(potential_heads) > 0:
                        head_idx = np.random.choice(potential_heads)
                    else:
                        head_idx = np.random.choice(cluster_nodes)
                    solution[head_idx] = 1

            fitness = self.calculate_fitness(solution)
            wolves.append({
                'position': solution.copy(),
                'fitness': fitness
            })

        # 按适应度降序排序
        wolves.sort(key=lambda x: x['fitness'], reverse=True)

        # 初始化领导狼
        alpha_wolf = wolves[0].copy()
        beta_wolf = wolves[1].copy()
        delta_wolf = wolves[2].copy()

        # 追踪收敛
        convergence = []

        # 主循环
        for iteration in range(max_iterations):
            # 更新a参数(从2线性递减到0)
            a = 2 * (1 - iteration / max_iterations)

            # 更新每只狼的位置
            for i in range(3, population_size):
                new_position = np.zeros(self.num_nodes, dtype=int)

                for j in range(self.num_nodes):
                    # 基于α、β、δ狼计算新位置
                    r1 = np.random.random()
                    r2 = np.random.random()

                    d_alpha = abs(2 * r1 * alpha_wolf['position'][j] - wolves[i]['position'][j])
                    d_beta = abs(2 * r1 * beta_wolf['position'][j] - wolves[i]['position'][j])
                    d_delta = abs(2 * r1 * delta_wolf['position'][j] - wolves[i]['position'][j])

                    x1 = alpha_wolf['position'][j] - a * d_alpha
                    x2 = beta_wolf['position'][j] - a * d_beta
                    x3 = delta_wolf['position'][j] - a * d_delta

                    # 取三个狼位置的平均影响
                    x_avg = (x1 + x2 + x3) / 3

                    # 二进制GWO: 使用S型转移函数将连续值转换为二进制
                    s = 1 / (1 + np.exp(-10 * (x_avg - 0.5)))

                    # 普通节点降低成为簇头的概率
                    if self.node_types[j] == 1:
                        s *= 0.8

                    # 基于概率更新位置
                    new_position[j] = 1 if np.random.random() < s else 0

                # 确保每个区域至少有一个簇头
                for cluster_id in range(4):
                    cluster_nodes = np.where(self.cluster_labels == cluster_id)[0]
                    if np.sum(new_position[cluster_nodes]) == 0:
                        potential_heads = cluster_nodes[self.node_types[cluster_nodes] == 2]
                        if len(potential_heads) > 0:
                            head_idx = np.random.choice(potential_heads)
                        else:
                            head_idx = np.random.choice(cluster_nodes)
                        new_position[head_idx] = 1

                # 计算新适应度
                new_fitness = self.calculate_fitness(new_position)

                # 更新狼的位置
                wolves[i]['position'] = new_position
                wolves[i]['fitness'] = new_fitness

            # 更新领导狼
            wolves.sort(key=lambda x: x['fitness'], reverse=True)
            alpha_wolf = wolves[0].copy()
            beta_wolf = wolves[1].copy()
            delta_wolf = wolves[2].copy()

            # 记录收敛
            convergence.append(alpha_wolf['fitness'])

            if iteration % 10 == 0 or iteration == max_iterations - 1:
                print(f"迭代 {iteration + 1}/{max_iterations}: 最佳适应度 = {alpha_wolf['fitness']:.4f}")

        # 获取最终聚类结果
        self.proposed_clusters = self.get_clustering_result(alpha_wolf['position'])

        return {
            'best_solution': alpha_wolf['position'],
            'convergence': convergence,
            'clusters': self.proposed_clusters
        }

    def kmeans_algorithm(self):
        """
        基于K-means的聚类算法(聚类组网策略)
        """
        print("运行聚类组网策略(K-means)...")

        # 1. 确定簇的数量(根据区域数量)
        n_clusters = 4

        # 2. 运行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.node_positions)

        # 3. 为每个簇选择簇头(选择离质心最近的潜在簇头节点)
        cluster_heads = np.zeros(self.num_nodes, dtype=int)

        for cluster_id in range(n_clusters):
            # 获取该簇的所有节点
            cluster_nodes = np.where(cluster_labels == cluster_id)[0]

            # 计算到质心的距离
            centroid = kmeans.cluster_centers_[cluster_id]
            distances_to_centroid = np.linalg.norm(self.node_positions[cluster_nodes] - centroid, axis=1)

            # 找出潜在簇头节点
            potential_heads = cluster_nodes[self.node_types[cluster_nodes] == 2]

            if len(potential_heads) > 0:
                # 如果有潜在簇头，选择最接近质心的
                head_distances = distances_to_centroid[np.isin(cluster_nodes, potential_heads)]
                best_idx = potential_heads[np.argmin(head_distances)]
            else:
                # 否则选择最接近质心的普通节点
                best_idx = cluster_nodes[np.argmin(distances_to_centroid)]

            cluster_heads[best_idx] = 1

        # 4. 获取聚类结果
        self.kmeans_clusters = self.get_clustering_result(cluster_heads)

        return {
            'best_solution': cluster_heads,
            'clusters': self.kmeans_clusters
        }

    def self_organized_algorithm(self):
        """
        自私自利的策略(自私组网策略)
        """
        print("运行自私组网策略...")

        # 1. 计算每个节点到UAV的距离
        uav_distances = self.distances_to_uav.copy()

        # 2. 每个区域选择距离UAV最近的节点作为簇头
        cluster_heads = np.zeros(self.num_nodes, dtype=int)

        for cluster_id in range(4):
            # 获取该区域的所有节点
            cluster_nodes = np.where(self.cluster_labels == cluster_id)[0]

            # 计算该区域节点到UAV的距离
            cluster_uav_distances = uav_distances[cluster_nodes]

            # 找出潜在簇头节点
            potential_heads = cluster_nodes[self.node_types[cluster_nodes] == 2]

            if len(potential_heads) > 0:
                # 如果有潜在簇头，选择距UAV最近的
                head_distances = uav_distances[potential_heads]
                best_idx = potential_heads[np.argmin(head_distances)]
            else:
                # 否则选择距UAV最近的普通节点
                best_idx = cluster_nodes[np.argmin(cluster_uav_distances)]

            cluster_heads[best_idx] = 1

        # 3. 获取聚类结果
        self.self_organized_clusters = self.get_clustering_result(cluster_heads)

        return {
            'best_solution': cluster_heads,
            'clusters': self.self_organized_clusters
        }

    def calculate_fitness(self, solution):
        """
        计算适应度函数
        """
        solution_binary = np.array(solution, dtype=bool)

        # 获取所有簇头索引
        cluster_heads = np.where(solution_binary)[0]

        if len(cluster_heads) == 0:
            return -np.inf  # 无簇头，返回极低适应度

        # 1. 上传性能
        distances_to_uav = self.distances_to_uav[cluster_heads]
        upload_performance = np.sum(1 / (1 + 0.01 * distances_to_uav))

        # 2. 簇内通信成本
        intra_cluster_cost = 0

        # 为每个非簇头节点找到最近的簇头
        non_ch_indices = np.where(~solution_binary)[0]

        if len(non_ch_indices) > 0:
            # 计算每个非簇头节点到最近簇头的距离
            min_distances = np.min(self.distance_matrix[non_ch_indices][:, cluster_heads], axis=1)
            intra_cluster_cost = np.sum(min_distances)

        # 3. 簇头数量平衡因子
        ch_count_factor = len(cluster_heads) / 4  # 理想情况下每个区域1个簇头
        ch_balance = 0

        for cluster_id in range(4):
            # 获取该区域的簇头数量
            cluster_nodes = np.where(self.cluster_labels == cluster_id)[0]
            ch_in_cluster = np.sum(solution_binary[cluster_nodes])

            # 如果该区域没有簇头，增加惩罚
            if ch_in_cluster == 0:
                ch_balance -= 100
            else:
                ch_balance += (1 - abs(1 - ch_in_cluster))  # 奖励每个区域正好1个簇头

        # 综合适应度计算
        fitness = (0.4 * upload_performance -
                   0.3 * intra_cluster_cost +
                   0.3 * ch_balance)

        return fitness

    def get_clustering_result(self, solution):
        """
        根据给定的簇头分配获取聚类结果
        """
        solution_binary = np.array(solution, dtype=bool)
        cluster_heads = np.where(solution_binary)[0]

        clusters = []

        for i, head_idx in enumerate(cluster_heads):
            head_pos = self.node_positions[head_idx]
            cluster_id = self.cluster_labels[head_idx]  # 簇头所在的原始区域

            clusters.append({
                'id': i,
                'head_idx': int(head_idx),
                'head_id': self.node_ids[head_idx],
                'head_pos': head_pos,
                'original_cluster': int(cluster_id),
                'members': [int(head_idx)]  # 初始仅包含簇头自身
            })

        # 为每个非簇头节点分配到最近的簇头
        non_ch_indices = np.where(~solution_binary)[0]

        for node_idx in non_ch_indices:
            # 计算到所有簇头的距离
            distances = self.distance_matrix[node_idx, cluster_heads]
            # 找到最近的簇头
            closest_ch = np.argmin(distances)
            # 将节点添加到相应的簇
            clusters[closest_ch]['members'].append(int(node_idx))

        return clusters

    def plot_clustering_comparison(self, save_path=None):
        """
        绘制三种聚类策略的对比图
        """
        if (self.proposed_clusters is None or
                self.kmeans_clusters is None or
                self.self_organized_clusters is None):
            print("错误: 请先运行所有聚类算法")
            return

        # 设置图表大小和布局
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # 设置子图标题
        titles = ['(a) 所提组网策略', '(b) 聚类组网策略', '(c) 自私组网策略']
        cluster_results = [self.proposed_clusters, self.kmeans_clusters, self.self_organized_clusters]

        # 生成聚类区域的颜色
        cluster_colors = ['cyan', 'blue', 'lime', 'orange']

        # 定义区域名称
        area_names = ['Co1', 'Co2', 'Co3', 'Co4']

        for i, (title, clusters) in enumerate(zip(titles, cluster_results)):
            ax = axs[i]

            # 绘制UAV轨迹
            uav_start = self.uav_trajectory['start']
            uav_end = self.uav_trajectory['end']
            ax.plot([uav_start[0], uav_end[0]], [uav_start[1], uav_end[1]],
                    'r--', linewidth=1, alpha=0.7)

            # 添加红色箭头表示UAV飞行方向
            arrow_y = 1200
            ax.arrow(500, arrow_y, 0, -50, head_width=30, head_length=30,
                     fc='r', ec='r', linewidth=1.5)

            # 绘制UAV图标
            uav_y = -100
            ax.scatter(500, uav_y, marker='s', s=100, color='brown', edgecolor='k')

            # 绘制簇区域
            for area_id in range(4):
                # 获取该区域的节点
                area_nodes = np.where(self.cluster_labels == area_id)[0]
                area_positions = self.node_positions[area_nodes]

                # 计算区域中心和半径
                center_x = np.mean(area_positions[:, 0])
                center_y = np.mean(area_positions[:, 1])

                # 计算椭圆参数
                width = np.std(area_positions[:, 0]) * 6
                height = np.std(area_positions[:, 1]) * 6

                # 绘制椭圆表示区域
                ellipse = Ellipse((center_x, center_y), width, height,
                                  fill=False, linestyle='--',
                                  edgecolor=cluster_colors[area_id],
                                  linewidth=1.5, alpha=0.7)
                ax.add_patch(ellipse)

                # 添加区域标签
                ax.text(center_x, center_y + height / 2 + 30, area_names[area_id],
                        ha='center', va='center', fontsize=10,
                        color=cluster_colors[area_id])

            # 绘制非簇头节点
            for cluster in clusters:
                head_idx = cluster['head_idx']
                members = cluster['members']
                original_cluster_id = cluster['original_cluster']
                color = cluster_colors[original_cluster_id]

                # 绘制成员节点(排除簇头)
                for member_idx in members:
                    if member_idx != head_idx:  # 非簇头节点
                        member_pos = self.node_positions[member_idx]
                        member_id = self.node_ids[member_idx]

                        # 绘制节点
                        ax.scatter(member_pos[0], member_pos[1], s=50,
                                   color=color, edgecolor='black',
                                   linewidth=0.5, alpha=0.7, zorder=2)

                        # 添加节点ID标签
                        ax.text(member_pos[0] + 10, member_pos[1] + 10, member_id,
                                fontsize=8, color='black')

                        # 绘制到簇头的连接线
                        head_pos = self.node_positions[head_idx]
                        ax.plot([member_pos[0], head_pos[0]], [member_pos[1], head_pos[1]],
                                '-', color=color, linewidth=0.8, alpha=0.5, zorder=1)

            # 绘制簇头节点
            for cluster in clusters:
                head_idx = cluster['head_idx']
                head_pos = self.node_positions[head_idx]
                head_id = self.node_ids[head_idx]
                original_cluster_id = cluster['original_cluster']
                color = cluster_colors[original_cluster_id]

                # 绘制簇头
                ax.scatter(head_pos[0], head_pos[1], s=100,
                           color=color, edgecolor='black',
                           linewidth=1.5, marker='*', zorder=3)

                # 添加簇头ID标签
                ax.text(head_pos[0] + 10, head_pos[1] + 10, head_id,
                        fontsize=8, fontweight='bold', color='black')

                # 绘制簇头到UAV的连接线
                # 计算簇头在UAV轨迹上的投影点
                start = np.array(self.uav_trajectory['start'])
                end = np.array(self.uav_trajectory['end'])
                trajectory_vec = end - start
                t = max(0, min(1, np.dot(head_pos - start, trajectory_vec) / np.sum(trajectory_vec ** 2)))
                projection = start + t * trajectory_vec

                # 绘制连接线(粗黑箭头表示)
                ax.plot([head_pos[0], projection[0]], [head_pos[1], projection[1]],
                        '-', color='black', linewidth=1.5, zorder=2)

                # 添加箭头
                dx = projection[0] - head_pos[0]
                dy = projection[1] - head_pos[1]
                arrow_len = np.sqrt(dx ** 2 + dy ** 2)
                if arrow_len > 0:
                    # 箭头起点移近簇头一点
                    arrow_x = head_pos[0] + dx * 0.3
                    arrow_y = head_pos[1] + dy * 0.3

                    # 归一化方向向量
                    dx /= arrow_len
                    dy /= arrow_len

                    # 绘制箭头
                    arrow_length = 50
                    ax.arrow(arrow_x, arrow_y, dx * arrow_length, dy * arrow_length,
                             head_width=20, head_length=20,
                             fc='black', ec='black', zorder=3)

            # 设置轴标签和范围
            ax.set_xlabel('X (×10² m)', fontsize=10)
            ax.set_ylabel('Y (×10² m)', fontsize=10)
            ax.set_xlim(-200, 1200)
            ax.set_ylim(-200, 1200)
            ax.set_title(title, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)

        # 添加总标题
        fig.suptitle('图 2 组网策略示意图', fontsize=14, y=0.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存至: {save_path}")

        plt.show()


# 主函数
def main():
    # 设置问题参数
    params = {
        'num_nodes': 40,  # 节点数量
        'area_size': 1000  # 区域大小
    }

    # 创建求解器实例
    optimizer = NetworkOptimizer(params)

    # 运行三种算法
    proposed_result = optimizer.proposed_algorithm()
    kmeans_result = optimizer.kmeans_algorithm()
    self_organized_result = optimizer.self_organized_algorithm()

    # 绘制对比图
    optimizer.plot_clustering_comparison(save_path="network_comparison.png")

    # 打印结果统计
    print("\n算法结果统计:")
    print(f"所提组网策略: {len(proposed_result['clusters'])} 个簇")
    print(f"聚类组网策略: {len(kmeans_result['clusters'])} 个簇")
    print(f"自私组网策略: {len(self_organized_result['clusters'])} 个簇")


if __name__ == "__main__":
    main()