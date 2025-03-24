import numpy as np
import matplotlib.pyplot as plt
import random
import time
import copy
import matplotlib

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


class MTSP:
    def __init__(self, num_cities, num_salesmen, city_positions=None):
        """初始化多旅行商问题"""
        self.num_cities = num_cities
        self.num_salesmen = num_salesmen

        # 如果没有提供城市坐标，则随机生成
        if city_positions is None:
            self.city_positions = np.random.rand(num_cities, 2) * 100
        else:
            self.city_positions = city_positions

        # 计算城市间距离矩阵
        self.distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    self.distance_matrix[i, j] = np.sqrt(
                        (self.city_positions[i, 0] - self.city_positions[j, 0]) ** 2 +
                        (self.city_positions[i, 1] - self.city_positions[j, 1]) ** 2
                    )

    def calculate_total_distance(self, tour):
        """计算总行程距离"""
        total_distance = 0

        # 找到分隔点（表示不同旅行商路径的分界点）
        salesmen_paths = self.get_salesmen_paths(tour)

        # 计算每个旅行商路径的距离
        for path in salesmen_paths:
            path_distance = 0
            # 计算路径中相邻城市之间的距离
            for i in range(len(path) - 1):
                path_distance += self.distance_matrix[path[i], path[i + 1]]
            # 添加返回起点的距离（如果路径不为空）
            if len(path) > 0:
                path_distance += self.distance_matrix[path[-1], path[0]]
            total_distance += path_distance

        return total_distance

    def get_salesmen_paths(self, tour):
        """将tour拆分为多个旅行商的路径"""
        # 计算每个旅行商大致需要访问的城市数量
        cities_per_salesman = self.num_cities // self.num_salesmen
        remainder = self.num_cities % self.num_salesmen

        paths = []
        start_idx = 0

        for i in range(self.num_salesmen):
            # 如果有余数，前remainder个旅行商多分配一个城市
            path_length = cities_per_salesman + (1 if i < remainder else 0)
            if path_length > 0:  # 确保路径长度大于0
                paths.append(tour[start_idx:start_idx + path_length])
                start_idx += path_length

        return paths

    def plot_tour(self, tour, title="MTSP Tour"):
        """绘制旅行商路径"""
        plt.figure(figsize=(10, 8))

        # 绘制所有城市点
        plt.scatter(self.city_positions[:, 0], self.city_positions[:, 1], c='blue', s=40)

        # 标记第一个城市（起点）
        plt.scatter(self.city_positions[0, 0], self.city_positions[0, 1], c='red', s=100, marker='*')

        # 为不同旅行商路径使用不同颜色
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

        # 获取每个旅行商的路径
        salesmen_paths = self.get_salesmen_paths(tour)

        # 绘制每个旅行商的路径
        for i, path in enumerate(salesmen_paths):
            color = colors[i % len(colors)]
            for j in range(len(path) - 1):
                plt.plot([self.city_positions[path[j], 0], self.city_positions[path[j + 1], 0]],
                         [self.city_positions[path[j], 1], self.city_positions[path[j + 1], 1]],
                         c=color, linestyle='-', linewidth=1)

            # 连接路径的最后一个城市到第一个城市（形成环路）
            if len(path) > 0:
                plt.plot([self.city_positions[path[-1], 0], self.city_positions[path[0], 0]],
                         [self.city_positions[path[-1], 1], self.city_positions[path[0], 1]],
                         c=color, linestyle='-', linewidth=1)

        plt.title(title)
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.grid(True)
        plt.show()


class GWO_MTSP:
    def __init__(self, mtsp, num_wolves=1000, max_iterations=1000, enable_2opt=True,
                 enable_init_optimization=True, enable_subgroup=True):
        """初始化灰狼算法"""
        self.mtsp = mtsp
        self.num_wolves = num_wolves
        self.max_iterations = max_iterations
        self.enable_2opt = enable_2opt
        self.enable_init_optimization = enable_init_optimization
        self.enable_subgroup = enable_subgroup

        # 用于记录收敛过程
        self.convergence_curve = np.zeros(max_iterations)

        # 初始化狼群（每个狼代表一个可能的解决方案）
        self.wolves = []
        self.initialize_wolves()

        # 初始化前三名狼（alpha, beta, delta）
        self.alpha_score = float('inf')
        self.alpha_pos = None
        self.beta_score = float('inf')
        self.beta_pos = None
        self.delta_score = float('inf')
        self.delta_pos = None

        # 如果启用子群算法，将狼群分成子群
        if self.enable_subgroup:
            self.num_subgroups = 3
            self.subgroups = self.divide_into_subgroups()

    def initialize_wolves(self):
        """初始化狼群"""
        for _ in range(self.num_wolves):
            # 初始化为随机解
            wolf = list(range(self.mtsp.num_cities))
            random.shuffle(wolf)

            # 如果启用初始解优化，应用贪心策略
            if self.enable_init_optimization:
                wolf = self.greedy_initialization()

            self.wolves.append(wolf)

    def greedy_initialization(self):
        """使用贪心策略创建一个较好的初始解"""
        tour = [0]  # 始终从城市0开始
        unvisited = list(range(1, self.mtsp.num_cities))

        while unvisited:
            last_city = tour[-1]
            # 找到距离最后一个城市最近的城市
            next_city = min(unvisited, key=lambda city: self.mtsp.distance_matrix[last_city, city])
            tour.append(next_city)
            unvisited.remove(next_city)

        return tour

    def divide_into_subgroups(self):
        """将狼群分成子群"""
        subgroups = [[] for _ in range(self.num_subgroups)]
        for i, wolf in enumerate(self.wolves):
            subgroup_idx = i % self.num_subgroups
            subgroups[subgroup_idx].append(i)
        return subgroups

    def apply_2opt(self, tour):
        """应用2-opt优化来改进路径"""
        # 对每个旅行商的路径分别应用2-opt
        paths = self.mtsp.get_salesmen_paths(tour)
        new_tour = []

        for path in paths:
            # 只有当路径长度大于3时，2-opt才有意义
            if len(path) > 3:
                improved = True
                while improved:
                    improved = False
                    for i in range(len(path) - 2):
                        for j in range(i + 2, len(path)):
                            # 计算当前路径距离
                            current_distance = (self.mtsp.distance_matrix[path[i], path[i + 1]] +
                                                self.mtsp.distance_matrix[path[j], path[(j + 1) % len(path)]])
                            # 计算交换后的路径距离
                            new_distance = (self.mtsp.distance_matrix[path[i], path[j]] +
                                            self.mtsp.distance_matrix[path[i + 1], path[(j + 1) % len(path)]])

                            # 如果交换后距离更短，应用2-opt
                            if new_distance < current_distance:
                                path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                                improved = True
            new_tour.extend(path)

        return new_tour

    def update_positions(self, iteration):
        """更新狼的位置"""
        a = 2 - iteration * (2 / self.max_iterations)  # 随着迭代减小

        for i in range(self.num_wolves):
            new_position = self.wolves[i].copy()

            # 针对灰狼算法的位置更新，我们需要适应离散问题
            # 使用交换操作来模拟位置更新
            # 从alpha, beta, delta学习
            if random.random() < 0.33:  # 学习自alpha
                r1 = random.random()
                A1 = 2 * a * r1 - a
                self.learn_from_leader(new_position, self.alpha_pos, abs(A1))
            elif random.random() < 0.66:  # 学习自beta
                r2 = random.random()
                A2 = 2 * a * r2 - a
                self.learn_from_leader(new_position, self.beta_pos, abs(A2))
            else:  # 学习自delta
                r3 = random.random()
                A3 = 2 * a * r3 - a
                self.learn_from_leader(new_position, self.delta_pos, abs(A3))

            # 应用2-opt优化
            if self.enable_2opt:
                new_position = self.apply_2opt(new_position)

            # 更新位置
            self.wolves[i] = new_position

            # 评估新位置
            fitness = self.mtsp.calculate_total_distance(new_position)

            # 更新alpha, beta, delta
            if fitness < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos
                self.alpha_score = fitness
                self.alpha_pos = new_position.copy()
            elif fitness < self.beta_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos
                self.beta_score = fitness
                self.beta_pos = new_position.copy()
            elif fitness < self.delta_score:
                self.delta_score = fitness
                self.delta_pos = new_position.copy()

    def learn_from_leader(self, position, leader_pos, intensity):
        """从领导者学习（适用于排列问题）"""
        if leader_pos is None:
            return

        # 根据intensity决定要交换的基因数量
        num_swaps = int(intensity * len(position) / 5) + 1
        num_swaps = min(num_swaps, len(position) // 2)

        # 随机选择一些位置从领导者那里学习
        for _ in range(num_swaps):
            idx = random.randint(0, len(position) - 1)
            leader_val = leader_pos[idx]

            # 找到当前位置中leader_val的位置
            current_idx = position.index(leader_val)

            # 交换这两个位置的值
            position[idx], position[current_idx] = position[current_idx], position[idx]

    def update_subgroups(self):
        """更新子群（每个子群有自己的领导者）"""
        # 为每个子群找到最佳狼
        for subgroup_idx, subgroup in enumerate(self.subgroups):
            best_wolf_idx = min(subgroup, key=lambda idx: self.mtsp.calculate_total_distance(self.wolves[idx]))
            best_wolf = self.wolves[best_wolf_idx]

            # 子群内其他狼向最佳狼学习
            for wolf_idx in subgroup:
                if wolf_idx != best_wolf_idx:
                    intensity = random.random() * 0.5  # 控制学习强度
                    self.learn_from_leader(self.wolves[wolf_idx], best_wolf, intensity)

    def optimize(self):
        """运行灰狼优化算法"""
        start_time = time.time()

        # 评估初始狼群
        for i, wolf in enumerate(self.wolves):
            fitness = self.mtsp.calculate_total_distance(wolf)

            # 更新alpha, beta, delta
            if fitness < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos
                self.beta_score = self.alpha_score
                self.beta_pos = self.alpha_pos
                self.alpha_score = fitness
                self.alpha_pos = wolf.copy()
            elif fitness < self.beta_score:
                self.delta_score = self.beta_score
                self.delta_pos = self.beta_pos
                self.beta_score = fitness
                self.beta_pos = wolf.copy()
            elif fitness < self.delta_score:
                self.delta_score = fitness
                self.delta_pos = wolf.copy()

        # 迭代优化
        for iteration in range(self.max_iterations):
            # 更新狼的位置
            self.update_positions(iteration)

            # 如果启用子群算法，更新子群
            if self.enable_subgroup:
                self.update_subgroups()

            # 记录收敛过程（使用alpha狼的适应度，这里是最短距离，所以取倒数使其变为最大化问题）
            self.convergence_curve[iteration] = 1 / self.alpha_score

            # 每10次迭代打印一次当前最佳解
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Best distance: {self.alpha_score:.2f}")

        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
        print(f"Best distance found: {self.alpha_score:.2f}")

        return self.alpha_pos, self.alpha_score, self.convergence_curve


def compare_algorithms(mtsp, num_wolves=30, max_iterations=100):
    """比较不同优化策略的灰狼算法"""
    # 完全优化版本
    print("Running full optimization GWO...")
    gwo_full = GWO_MTSP(mtsp, num_wolves, max_iterations,
                        enable_2opt=True, enable_init_optimization=True, enable_subgroup=True)
    best_tour_full, best_distance_full, convergence_full = gwo_full.optimize()

    # 无2-opt优化版本
    print("\nRunning GWO without 2-opt...")
    gwo_no_2opt = GWO_MTSP(mtsp, num_wolves, max_iterations,
                           enable_2opt=False, enable_init_optimization=True, enable_subgroup=True)
    best_tour_no_2opt, best_distance_no_2opt, convergence_no_2opt = gwo_no_2opt.optimize()

    # 无初始化优化版本
    print("\nRunning GWO without initial optimization...")
    gwo_no_init = GWO_MTSP(mtsp, num_wolves, max_iterations,
                           enable_2opt=True, enable_init_optimization=False, enable_subgroup=True)
    best_tour_no_init, best_distance_no_init, convergence_no_init = gwo_no_init.optimize()

    # 无子群算法版本
    print("\nRunning GWO without subgroup optimization...")
    gwo_no_subgroup = GWO_MTSP(mtsp, num_wolves, max_iterations,
                               enable_2opt=True, enable_init_optimization=True, enable_subgroup=False)
    best_tour_no_subgroup, best_distance_no_subgroup, convergence_no_subgroup = gwo_no_subgroup.optimize()

    # 绘制收敛对比图
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_full, label='Full Optimization')
    plt.plot(convergence_no_2opt, label='Without 2-opt')
    plt.plot(convergence_no_init, label='Without Init Optimization')
    plt.plot(convergence_no_subgroup, label='Without Subgroup')
    plt.title('Convergence Comparison of GWO with Different Optimizations')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness (1/Total Distance)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 打印最佳结果
    print("\n比较结果:")
    print(f"完全优化: {best_distance_full:.2f}")
    print(f"无2-opt优化: {best_distance_no_2opt:.2f}")
    print(f"无初始化优化: {best_distance_no_init:.2f}")
    print(f"无子群优化: {best_distance_no_subgroup:.2f}")

    # 画出最佳路线图
    mtsp.plot_tour(best_tour_full, "Best Tour (Full Optimization)")

    return best_tour_full, best_distance_full


# 主函数
if __name__ == "__main__":
    # 设置问题参数
    num_cities = 50
    num_salesmen = 1

    # 创建MTSP实例
    mtsp = MTSP(num_cities, num_salesmen)

    # 比较不同优化策略
    best_tour, best_distance = compare_algorithms(mtsp, num_wolves=50, max_iterations=100)