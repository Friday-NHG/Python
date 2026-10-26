import gymnasium as gym  # 强化学习环境库，用于创建和运行 CartPole 游戏环境
import math  # 数学函数；这里用于计算探索概率的指数衰减
import random  # 随机数工具；用于随机选动作和抽取经验样本

import matplotlib  # 绘图库主模块；这里用于设置绘图后端
matplotlib.use("TkAgg")  # 指定 TkAgg 后端，用于弹出独立绘图窗口
import matplotlib.pyplot as plt  # 绘图接口；用于显示每回合时长和训练曲线

from collections import namedtuple, deque  # namedtuple 定义经验记录；deque 保存有限容量的经验回放
from itertools import count  # 生成步数计数，用于逐步运行每个回合

import torch  # PyTorch 主库；用于张量、设备选择和模型计算
import torch.nn as nn  # 神经网络模块；用于定义网络层和损失函数
import torch.optim as optim  # 优化器模块；用于更新神经网络参数
import torch.nn.functional as F  # 神经网络函数接口；这里用于 ReLU 激活函数

env = gym.make("CartPole-v1", render_mode="human")  # 创建 CartPole 环境，CartPole-v1 是“小车倒立摆”任务：智能体每一步观察环境状态，然后选择把小车向左或向右移动。
# render_mode="human" 表示运行时尝试显示游戏画面

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# 定义一条训练经验 包含：
# - state：执行动作之前的状态。
# - action：智能体采取的动作。
# - next_state：执行动作后到达的状态。
# - reward：这一步得到的奖励
# DQN 会使用这些经历来学习“在什么状态采取什么动作比较好”
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# ReplayMemory 是经验回放池。它内部用 deque 保存经验，最多存放 capacity 条；满了以后，新的经验会替换掉较旧的经验
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 定义一个全连接神经网络：
# - 输入层接收状态数值。CartPole 有 4 个状态值，所以输入层有 4 个输入。
# - 第一层把输入转换为 128 个数值。
# - 第二层仍输出 128 个数值。
# - 输出层的数值个数等于可选动作数。CartPole 有 2 个动作，因此输出 2 个值。
# 这两个输出分别代表网络估计的“向左”和“向右”的 Q 值。通常，Q 值越大，代表网络认为该动作带来的长期回报越高。
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# BATCH_SIZE = 128：每次从经验回放池抽取 128 条经验进行训练。
# GAMMA = 0.99：折扣因子。越接近 1，模型越重视未来奖励。
# EPS_START = 0.9：训练开始时的探索概率较高，初始约为 90%。
# EPS_END = 0.01：探索概率逐渐下降，最低约为 1%。
# EPS_DECAY = 2500：控制探索概率下降的速度；值越大，下降越慢。
# TAU = 0.005：目标网络每次向策略网络靠近的比例。
# LR = 3e-4：学习率，决定每次更新网络参数的幅度。
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


# 获取可选动作数。CartPole 中通常是2
n_actions = env.action_space.n
# 重置环境，开始一个新回合，并返回初始状态和额外信息。这里的 state 是初始状态，info 是环境附带的信息，本程序没有继续使用它。
state, info = env.reset()
n_observations = len(state)


# 创建策略网络、目标网络和经验池
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
# 创建 AdamW 优化器，用来更新 policy_net 的参数。经验池最多保存 10,000 条经验
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


# 根据 Q 值选择动作
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
