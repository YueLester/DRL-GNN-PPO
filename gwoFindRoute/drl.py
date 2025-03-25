import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random


# 网络环境模拟器 (简化版)
class NetworkEnv:
    def __init__(self, num_nodes, num_tasks, time_horizon):
        self.num_nodes = num_nodes
        self.num_tasks = num_tasks
        self.time_horizon = time_horizon
        self.current_time = 0

        # 初始化网络状态
        self.reset()

    def reset(self):
        # 初始化节点容量
        self.node_capacities = np.random.randint(100, 200, size=self.num_nodes)

        # 当前节点负载
        self.node_loads = np.zeros(self.num_nodes)

        # 任务队列 - 每个任务有大小、优先级、截止时间
        self.task_queue = []
        for _ in range(self.num_tasks):
            self.task_queue.append({
                'size': np.random.randint(10, 50),
                'priority': np.random.randint(1, 5),
                'deadline': np.random.randint(1, self.time_horizon)
            })

        self.completed_tasks = []
        self.current_time = 0

        return self._get_high_level_state()

    def _get_high_level_state(self):
        """返回高层策略的状态表示"""
        # 网络负载概况
        avg_load = np.mean(self.node_loads / self.node_capacities)
        load_std = np.std(self.node_loads / self.node_capacities)

        # 任务队列状态
        avg_size = np.mean([t['size'] for t in self.task_queue]) if self.task_queue else 0
        avg_priority = np.mean([t['priority'] for t in self.task_queue]) if self.task_queue else 0
        queue_length = len(self.task_queue)

        # 时间信息
        time_progress = self.current_time / self.time_horizon

        return np.array([
            avg_load, load_std, avg_size, avg_priority,
            queue_length, time_progress
        ])

    def _get_mid_level_state(self):
        """返回中层策略的状态表示"""
        # 节点状态
        node_loads_norm = self.node_loads / self.node_capacities

        # 当前任务状态 (假设中层处理前5个任务)
        task_features = []
        for i in range(min(5, len(self.task_queue))):
            task = self.task_queue[i]
            task_features.append([
                task['size'] / 50,  # 归一化大小
                task['priority'] / 5,  # 归一化优先级
                task['deadline'] / self.time_horizon  # 归一化截止时间
            ])

        # 填充，确保有固定数量的任务特征
        while len(task_features) < 5:
            task_features.append([0, 0, 0])

        return {
            'node_status': node_loads_norm,
            'task_features': np.array(task_features)
        }

    def _get_low_level_state(self, node_id, task_id):
        """返回低层策略的状态表示"""
        if task_id >= len(self.task_queue):
            return np.zeros(5)  # 无效任务

        task = self.task_queue[task_id]
        node_load = self.node_loads[node_id] / self.node_capacities[node_id]

        return np.array([
            node_load,  # 节点负载率
            task['size'] / 50,  # 任务大小
            task['priority'] / 5,  # 任务优先级
            task['deadline'] / self.time_horizon,  # 任务截止时间
            (self.node_capacities[node_id] - self.node_loads[node_id]) / self.node_capacities[node_id]  # 节点剩余容量
        ])

    def step_high_level(self, action):
        """高层动作：设定资源分配策略
        action: 各节点的资源分配权重
        """
        # 归一化权重
        allocation_weights = action / np.sum(action)

        # 设定资源分配策略，这将影响中层的任务分配
        self.allocation_strategy = allocation_weights

        # 高层策略每步前进多个时间单位
        high_level_steps = 5
        rewards = []

        for _ in range(high_level_steps):
            if self.current_time >= self.time_horizon:
                break

            # 执行中层策略 (模拟)
            mid_reward = self._simulate_mid_level()
            rewards.append(mid_reward)

            # 更新时间
            self.current_time += 1

        done = self.current_time >= self.time_horizon

        # 高层奖励：中层奖励的累积 + 负载均衡度
        load_balance_score = -np.std(self.node_loads / self.node_capacities) * 10
        high_reward = sum(rewards) + load_balance_score

        return self._get_high_level_state(), high_reward, done, {
            'load_balance': load_balance_score,
            'completed_tasks': len(self.completed_tasks)
        }

    def _simulate_mid_level(self):
        """模拟中层策略执行"""
        # 根据高层分配策略，为每个节点确定可分配的任务数
        node_task_capacity = np.ceil(self.allocation_strategy * min(5, len(self.task_queue)))
        node_tasks_assigned = np.zeros(self.num_nodes, dtype=int)

        # 处理任务 (简化：按优先级降序)
        sorted_tasks = sorted(
            enumerate(self.task_queue),
            key=lambda x: x[1]['priority'],
            reverse=True
        )

        total_reward = 0
        tasks_to_remove = []

        for task_idx, task in sorted_tasks:
            if len(tasks_to_remove) >= 5:  # 最多处理5个任务
                break

            # 选择最适合的节点 (负载最低且有分配额度的节点)
            eligible_nodes = [
                n for n in range(self.num_nodes)
                if node_tasks_assigned[n] < node_task_capacity[n] and
                   self.node_loads[n] + task['size'] <= self.node_capacities[n]
            ]

            if eligible_nodes:
                # 选择负载最低的节点
                node_id = min(eligible_nodes, key=lambda n: self.node_loads[n] / self.node_capacities[n])

                # 执行低层操作 (简化)
                success, low_reward = self._execute_low_level(node_id, task)

                if success:
                    # 更新节点负载
                    self.node_loads[node_id] += task['size']
                    node_tasks_assigned[node_id] += 1

                    # 记录任务完成
                    tasks_to_remove.append(task_idx)
                    self.completed_tasks.append(task)

                    # 累积奖励
                    total_reward += low_reward

        # 移除已完成的任务
        self.task_queue = [t for i, t in enumerate(self.task_queue) if i not in tasks_to_remove]

        # 添加新任务
        new_tasks = max(0, np.random.poisson(2) - len(tasks_to_remove))
        for _ in range(new_tasks):
            self.task_queue.append({
                'size': np.random.randint(10, 50),
                'priority': np.random.randint(1, 5),
                'deadline': self.current_time + np.random.randint(1, 10)
            })

        # 检查任务截止时间，移除过期任务并施加惩罚
        expired_tasks = [i for i, t in enumerate(self.task_queue) if t['deadline'] <= self.current_time]
        if expired_tasks:
            # 每个过期任务的惩罚与其优先级成正比
            expired_penalty = sum(self.task_queue[i]['priority'] * 5 for i in expired_tasks)
            total_reward -= expired_penalty

            # 移除过期任务
            self.task_queue = [t for i, t in enumerate(self.task_queue) if i not in expired_tasks]

        return total_reward

    def _execute_low_level(self, node_id, task):
        """模拟低层执行结果"""
        # 简化模型：根据节点负载和任务特性计算成功概率
        remaining_capacity = self.node_capacities[node_id] - self.node_loads[node_id]
        success_prob = min(1.0, remaining_capacity / task['size']) * 0.9

        success = random.random() < success_prob

        if success:
            # 计算奖励：任务优先级 * 完成及时性
            timeliness = 1.0 - (self.current_time / task['deadline']) if task['deadline'] > 0 else 0
            reward = task['priority'] * (1 + timeliness)
        else:
            # 失败惩罚
            reward = -task['priority'] * 0.5

        return success, reward


# 高层代理 - 负责资源分配策略
class HighLevelAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_dim, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # 探索：返回随机分配策略
            return np.random.random(self.action_dim)

        # 利用：使用模型预测最佳策略
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.model.predict(state, verbose=0)[0]
        return action_probs

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # 随机抽取批次
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        # 计算目标Q值
        target = rewards + self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1) * (1 - dones)

        # 获取当前Q值估计
        target_f = self.model.predict(states, verbose=0)

        # 更新目标Q值
        for i, action in enumerate(actions):
            # 简化：使用最大动作分量的索引
            action_idx = np.argmax(action)
            target_f[i][action_idx] = target[i]

        # 训练模型
        self.model.fit(states, target_f, epochs=1, verbose=0)

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


# 演示分层强化学习训练过程
def train_hierarchical_rl():
    # 环境参数
    num_nodes = 5
    num_tasks = 20
    time_horizon = 100

    # 创建环境
    env = NetworkEnv(num_nodes, num_tasks, time_horizon)

    # 创建高层代理
    high_level_state_dim = 6  # 高层状态维度
    high_level_action_dim = num_nodes  # 高层动作维度：节点资源分配权重
    high_agent = HighLevelAgent(high_level_state_dim, high_level_action_dim)

    # 训练参数
    num_episodes = 200

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 高层决策
            action = high_agent.act(state)

            # 执行高层动作
            next_state, reward, done, info = env.step_high_level(action)

            # 记忆经验
            high_agent.remember(state, action, reward, next_state, done)

            # 更新状态
            state = next_state
            total_reward += reward

            # 训练高层代理
            high_agent.replay()

        # 定期更新目标网络
        if episode % 10 == 0:
            high_agent.update_target_model()

        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, " +
              f"Completed Tasks: {info['completed_tasks']}, Epsilon: {high_agent.epsilon:.2f}")

# train_hierarchical_rl()  # 取消注释来执行训练
