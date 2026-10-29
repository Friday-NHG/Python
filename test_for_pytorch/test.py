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
    next_state = (nr, nc)
    if next_state == GOAL:
        return next_state, 10, True     # 到达终点
    return next_state, -1, False        # 每走一步惩罚

# ---------- 2. 初始化 Q 表 ----------
# Q[state][action] = 在 state 做 action 的“价值”
Q = {}
for r in range(SIZE):
    for c in range(SIZE):
        Q[(r, c)] = [0.0, 0.0, 0.0, 0.0]

# ---------- 3. 超参数 ----------
alpha = 0.1       # 学习率
gamma = 0.9       # 折扣因子
epsilon = 0.2     # 探索率
episodes = 1000   # 训练轮数

# ---------- 4. 训练循环 ----------
for ep in range(episodes):
    state = START
    done = False
    while not done:
        # (a) 选择动作：ε-贪心
        if random.random() < epsilon:
            a = random.randint(0, 3)              # 探索
        else:
            a = int(np.argmax(Q[state]))          # 利用

        # (b) 执行动作，观察结果
        next_state, reward, done = step(state, ACTIONS[a])

        # (c) Q-Learning 更新公式
        # Q(s,a) ← Q(s,a) + α [r + γ·max Q(s',a') − Q(s,a)]
        best_next = max(Q[next_state])
        Q[state][a] += alpha * (reward + gamma * best_next - Q[state][a])

        # (d) 进入下一状态
        state = next_state

# ---------- 5. 测试：按学到的策略走 ----------
state = START
path = [state]
while state != GOAL:
    a = int(np.argmax(Q[state]))   # 不再探索，纯利用
    state, _, _ = step(state, ACTIONS[a])
    path.append(state)
print("学到的最短路径：", path)