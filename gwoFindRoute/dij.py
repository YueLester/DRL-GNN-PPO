import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
from collections import defaultdict

# 设置随机种子以保证结果可重现
np.random.seed(42)


def generate_network(num_nodes=100, connection_radius=0.2):
    """
    生成一个随机网络，包含指定数量的节点

    参数:
    - num_nodes: 网络中的节点数量
    - connection_radius: 两个节点之间能够建立连接的最大距离

    返回:
    - positions: 节点位置的字典
    - graph: 网络图（邻接列表表示）
    - distances: 节点间的距离字典
    """
    # 在单位正方形内生成随机位置
    x = np.random.rand(num_nodes)
    y = np.random.rand(num_nodes)

    positions = {i: (x[i], y[i]) for i in range(num_nodes)}
    graph = defaultdict(list)
    distances = {}

    # 构建网络连接
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # 计算欧氏距离
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

            # 如果距离小于指定值，则建立连接
            if dist < connection_radius:
                graph[i].append(j)
                graph[j].append(i)
                distances[(i, j)] = distances[(j, i)] = dist

    return positions, graph, distances


def dijkstra(graph, distances, start, end):
    """
    使用Dijkstra算法找出从start到end的最短路径

    参数:
    - graph: 网络图
    - distances: 节点间的距离
    - start: 起始节点
    - end: 目标节点

    返回:
    - path: 最短路径
    - total_distance: 路径总长度
    """
    # 初始化
    dist = {node: float('infinity') for node in graph}
    dist[start] = 0
    previous = {node: None for node in graph}
    priority_queue = [(0, start)]
    visited = set()

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == end:
            break

        for neighbor in graph[current_node]:
            if neighbor in visited:
                continue

            edge = (current_node, neighbor) if (current_node, neighbor) in distances else (neighbor, current_node)
            weight = distances[edge]
            distance = current_dist + weight

            if distance < dist[neighbor]:
                dist[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # 重建路径
    path = []
    current = end

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()

    if path[0] != start:  # 如果无法到达终点
        return [], float('infinity')

    return path, dist[end]


def find_diverse_paths(graph, distances, positions, start, end, num_paths=3, diversity_weight=0.5):
    """
    找出多条多样化的路径

    参数:
    - graph: 网络图
    - distances: 节点间的距离
    - positions: 节点位置
    - start: 起始节点
    - end: 目标节点
    - num_paths: 要找的路径数量
    - diversity_weight: 多样性权重，控制路径多样性的程度

    返回:
    - paths: 发现的路径列表
    """
    paths = []
    # 首先找到最短路径
    first_path, first_dist = dijkstra(graph, distances, start, end)

    if not first_path:
        print(f"无法从节点 {start} 到达节点 {end}")
        return []

    paths.append((first_path, first_dist))

    # 修改后的网络，增加已用边的权重以鼓励多样性
    modified_distances = distances.copy()

    # 寻找额外的路径
    for i in range(1, num_paths):
        # 增加先前路径中使用的边的权重
        for path, _ in paths:
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                edge = (u, v) if (u, v) in modified_distances else (v, u)
                modified_distances[edge] *= (1 + diversity_weight)

        # 使用修改后的距离找到新路径
        new_path, new_dist = dijkstra(graph, modified_distances, start, end)

        if not new_path:
            print(f"无法找到第 {i + 1} 条从节点 {start} 到达节点 {end} 的路径")
            break

        paths.append((new_path, new_dist))

    return paths


def visualize_network_and_paths(positions, graph, paths):
    """
    可视化网络和发现的路径

    参数:
    - positions: 节点位置
    - graph: 网络图
    - paths: 发现的路径列表
    """
    # 创建NetworkX图
    G = nx.Graph()

    # 添加节点和边
    for node in graph:
        G.add_node(node)

    for node in graph:
        for neighbor in graph[node]:
            if node < neighbor:  # 避免重复添加
                G.add_edge(node, neighbor)

    # 创建图形
    plt.figure(figsize=(12, 10))

    # 绘制网络位置图
    plt.subplot(1, 2, 1)
    nx.draw(G, positions, node_size=50, node_color='lightblue',
            with_labels=False, alpha=0.7)

    # 突出显示起始和结束节点
    start_node = paths[0][0][0]
    end_node = paths[0][0][-1]
    nx.draw_networkx_nodes(G, positions, nodelist=[start_node],
                           node_color='green', node_size=100)
    nx.draw_networkx_nodes(G, positions, nodelist=[end_node],
                           node_color='red', node_size=100)

    # 标注起始和结束节点
    nx.draw_networkx_labels(G, positions,
                            labels={start_node: f"Start ({start_node})",
                                    end_node: f"End ({end_node})"},
                            font_size=10)

    plt.title("网络节点位置图")

    # 绘制链路示意图
    plt.subplot(1, 2, 2)
    nx.draw(G, positions, node_size=30, node_color='lightgray',
            with_labels=False, alpha=0.3, edge_color='lightgray')

    # 绘制发现的路径
    colors = ['red', 'blue', 'green']
    for i, (path, dist) in enumerate(paths):
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_nodes(G, positions, nodelist=path,
                               node_color=colors[i], node_size=60, alpha=0.7)
        nx.draw_networkx_edges(G, positions, edgelist=path_edges,
                               edge_color=colors[i], width=2, alpha=0.7)

    # 突出显示起始和结束节点
    nx.draw_networkx_nodes(G, positions, nodelist=[start_node],
                           node_color='green', node_size=100)
    nx.draw_networkx_nodes(G, positions, nodelist=[end_node],
                           node_color='red', node_size=100)

    # 标注起始和结束节点
    nx.draw_networkx_labels(G, positions,
                            labels={start_node: f"Start ({start_node})",
                                    end_node: f"End ({end_node})"},
                            font_size=10)

    # 添加路径信息
    legend_text = []
    for i, (path, dist) in enumerate(paths):
        legend_text.append(f"Path {i + 1}: {' -> '.join(map(str, path))} (dist: {dist:.3f})")

    plt.title("发现的链路示意图")
    plt.figtext(0.5, 0.01, '\n'.join(legend_text), ha='center', fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("network_paths.png", dpi=300)
    plt.show()


# 主程序
def main():
    # 生成网络
    positions, graph, distances = generate_network(num_nodes=100, connection_radius=0.2)

    # 确保网络是连通的
    connected_nodes = set()

    def dfs(node):
        connected_nodes.add(node)
        for neighbor in graph[node]:
            if neighbor not in connected_nodes:
                dfs(neighbor)

    if graph:  # 确保图不为空
        dfs(next(iter(graph)))

    print(f"网络中有 {len(connected_nodes)} 个连通节点，共 {len(graph)} 个节点")

    # 选择起始和结束节点
    # 选择距离较远的节点作为起始和结束点
    max_dist = 0
    start_node, end_node = 0, 0

    for i in connected_nodes:
        for j in connected_nodes:
            if i != j:
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist > max_dist:
                    max_dist = dist
                    start_node, end_node = i, j

    print(f"选择的起始节点: {start_node}, 结束节点: {end_node}")

    # 寻找多条路径
    paths = find_diverse_paths(graph, distances, positions, start_node, end_node, num_paths=3)

    if paths:
        # 打印路径信息
        print("\n发现的路径:")
        for i, (path, dist) in enumerate(paths):
            print(f"路径 {i + 1}: {' -> '.join(map(str, path))}")
            print(f"路径长度: {dist:.4f}\n")

        # 可视化网络和路径
        visualize_network_and_paths(positions, graph, paths)
    else:
        print("未能找到路径，请尝试增加连接半径或使用不同的起始/结束节点")


if __name__ == "__main__":
    main()
