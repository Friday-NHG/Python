import torch
import torch.nn as nn
import torch.optim as optim
# 智能体看状态 → 选动作 → 环境给奖励 → 存下这条经验
# → 攒一批经验 → 算误差 → 改网络参数 → 再玩

# ============================================================
# 0. 环境与网络定义
# ============================================================
torch.manual_seed(0)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 2)
        )
    def forward(self, s):
        return self.net(s)

q_net = QNet()
q_target_net = QNet()
q_target_net.load_state_dict(q_net.state_dict())   # 初始同步
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

# ============================================================
# 1. 从 replay buffer 采样一批 transition（“数据”）
#    形状：[batch=3, ...]
# ============================================================
states = torch.randn(3, 4)                    # [3, 4]
actions = torch.tensor([0, 1, 0])              # [3]
rewards = torch.tensor([1.0, 0.5, -0.2])       # [3]
next_states = torch.randn(3, 4)                    # [3, 4]
dones = torch.tensor([0.0, 1.0, 0.0])        # [3]

# ============================================================
# 2. 前向传播：网络对每个状态输出两个动作的 Q 值
#    q_values 带 grad_fn，梯度会穿过它
# ============================================================
q_values = q_net(states)                           # [3, 2]
# 例如 tensor([[0.12,-0.05],[0.30,0.18],[-0.08,0.22]], grad_fn=...)

# ============================================================
# 3. 构造 TD 目标（target 网络，no_grad → 梯度到此为止）
# ============================================================
with torch.no_grad():
    next_q = q_target_net(next_states)             # [3, 2]
    max_next_q = next_q.max(dim=1).values          # [3]
    td_target = rewards + 0.99 * (1 - dones) * max_next_q   # [3]

# 取出实际执行动作对应的 Q 值
q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [3]

# ============================================================
# 4. 算 loss 并汇总成一个标量（关键：backward 只能从标量出发）
# ============================================================
td_errors = (q_sa - td_target) ** 2                # [3] 每条经验一个误差
loss = td_errors.mean()                            # 标量 ← 一个数
print("loss =", loss)                              # tensor(..., grad_fn=<MeanBackward0>)

# ============================================================
# 5. backward：从标量倒着算梯度，落回 q_net 的参数
#    规则：标量张量可直接 backward，不用传 gradient
# ============================================================
loss.backward()

# 验证：梯度形状和参数完全一致；target 网络没有梯度
print("\n--- q_net 参数梯度 ---")
for name, p in q_net.named_parameters():
    print(f"{name:15s} shape={tuple(p.grad.shape)}  grad_norm={p.grad.norm():.4f}")

print("\n--- q_target_net 梯度 ---")
print(q_target_net.net[0].weight.grad)             # None

# ============================================================
# 6. 梯度下降：用梯度更新参数，然后清空梯度
# ============================================================
optimizer.step()        # 沿梯度反方向更新 q_net
optimizer.zero_grad()   # 清空 .grad，下一轮重新算

# ============================================================
# 7. target 网络软更新（不参与梯度，只是参数同步）
# ============================================================
tau = 0.005
with torch.no_grad():
    for tp, p in zip(q_target_net.parameters(), q_net.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)