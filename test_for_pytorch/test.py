import numpy as np
import random

# ---------- 1. 定义环境 ----------
SIZE = 5
START = (0, 0)
GOAL = (SIZE - 1, SIZE - 1)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上 下 左 右

def step(state, action):
    """执行动作，返回 (下一状态, 奖励, 是否结束)"""
    r, c = state
    dr, dc = action
    nr, nc = r + dr, c + dc
    # 越界就停在原地
    if not (0 <= nr < SIZE and 0 <= nc < SIZE):
        nr, nc = r, c
    next_state = (nr, nc)  # 计算下一个状态
    if next_state == GOAL:
        return next_state, 10, True     # 到达终点
    return next_state, -1, False        # 每走一步惩罚

# ---------- 2. 初始化 Q 表 ----------
# Q[state][action] = 在 state 做 action 的“最优动作价值估计”
# 理论上，Q∗(s,a)（真实最优价值）是"从 s 出发做 a，之后一直按最优策略走，能拿到的期望折扣总回报"
# 这个值要靠无数次采样求期望才能逼近，实际中永远算不出精确值
Q = {}  # 这是个字典
for r in range(SIZE):
    for c in range(SIZE):
        Q[(r, c)] = [0.0, 0.0, 0.0, 0.0]  # 这就是最初对Q值的估计，完全是瞎猜，后面的每次更新不断用新经验修正这个估计，让它越来越接近真实价值

# ---------- 3. 超参数 ----------
alpha = 0.1       # 学习率    学习率控制"每次更新时，旧 Q 值向新目标靠拢多大一步"——也就是智能体接受新经验的快慢。学得快但可能不稳，学得慢但更平滑
gamma = 0.9       # 折扣因子
epsilon = 0.2     # 初始探索率
episodes = 1000   # 训练轮数

# ---------- 4. 训练循环 ----------
for ep in range(episodes):
    state = START
    done = False
    epsilon = max(0.01, epsilon * 0.995)  # 探索率按指数衰减，前期快后期慢 每回合乘上0.995
    while not done:
        # (1) 选择动作：ε-贪心  a是动作的下标
        if random.random() < epsilon:  # random.random()输出0-1上的均匀分布，配合探索率决定探索还是利用
            a = random.randint(0, 3)  # 探索    输出闭区间0-3上的随机整数作为动作下标
        else:
            a = int(np.argmax(Q[state]))    # 利用    调用numpy库，返回Q表state位置上q值最大动作的下标

        # (2) 执行动作，观察结果
        next_state, reward, done = step(state, ACTIONS[a])

        # (3) Q-Learning 的标准更新公式（时序差分 TD 更新）
        # 旧估计 Q(s, a)
        #    ↓
        # 新目标 = 真实即时奖励 r + 折扣后的"下一状态最好价值"
        #    ↓
        # 误差 = 新目标 - 旧估计
        #    ↓
        # 新Q = 旧 Q + α × 误差   （朝着目标挪一小步）
        # Q(s,a) ← Q(s,a) + α [r + γ·max Q(s',a') − Q(s,a)]
        best_next = max(Q[next_state])  # 根据贝尔曼最优公式，下一状态的最好价值就是 max Q(s',a')
        Q[state][a] += alpha * (reward + gamma * best_next - Q[state][a])
        # 这个更新是"自举（bootstrapping）"的，"r + γ·max Q(s',a')" 中的 max Q(s',a') 本身也是一个估计值
        # 用下一状态的 Q 去修正当前状态的 Q

        # (4) 进入下一状态
        state = next_state

# ---------- 5. 测试：按学到的策略走 ----------
state = START
path = [state]
while state != GOAL:
    a = int(np.argmax(Q[state]))   # 最终的Q表就是训练成果，查Q表决定怎么行动
    state, _, _ = step(state, ACTIONS[a])
    path.append(state)
print("学到的最短路径：", path)
print(Q.values())