import numpy as np
import copy
# """创建环境"""
SIZE = 5
START = (0, 0)
END = (SIZE-1, SIZE-1)
BARRIER = [(2, 0), (1, 2), (0, 4), (4, 2)]
ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # 原地不动、上、下、左、右
# """定义参数"""
gamma = 0.9  # 折扣因子
the_number = 0.00001  #

# """初始化V值和策略policy"""
V = {}
policy = {}
for row in range(SIZE):
    for col in range(SIZE):
        V[(row, col)] = 0
        policy[(row, col)] = 0

# """定义行动规则"""
def step(st, act):
    ro, co = st
    dr, dc = ACTIONS[act]
    nr, nc = ro + dr, co + dc
    # 按action行动，撞墙留在原地
    if nr < 0 or nr >= SIZE or nc < 0 or nc >= SIZE:
        next_st = (ro, co)
    else:
        next_st = (ro + dr, co + dc)
    # 定义reward: 不进禁区就是 -1，进了 -2 ，进入target区域 +10
    if next_st in BARRIER:
        rew = -2
    elif next_st == END:
        rew = 10
    else:
        rew = -1
    return next_st, rew  # 返回下一状态和reward


# 循环更新V值以及策略
while True:
    V_pre = copy.deepcopy(V)  # 记录当前的V值
    # """遍历每个状态"""
    for state in V:
        if state == END:
            continue  # 终点不更新
        Q_list = []  # 创建空列表收集每个状态的q值
        for act in range(SIZE):  # 遍历每个动作
            next_state, reward = step(state, act)
            Q = reward + gamma * V[next_state]
            Q_list.append(Q)
        action = np.argmax(Q_list)
        policy[state] = action  # 更新策略
        V[state] = Q_list[action]  # 更新V值
    # """算更新前后V值的差"""
    diff = np.abs(np.array(list(V.values())) - np.array(list(V_pre.values())))
    if max(diff) < the_number:
        break

print('最优状态值表:', V)
print(policy)
# 测试
state = START
li = [state]
done = False
while not done:
    action = policy[state]
    print(state, action)
    next_state, reward = step(state, action)
    li.append(next_state)
    if next_state == END:
        done = True
    state = next_state
print('轨迹:', li)


