import torch
# 梯度: 函数在某点处变化(增长)最快的那个方向
# 在那个点处对所有维度求偏导再组成一个向量，这个向量就是变化最快的方向
# 梯度下降: 沿着梯度的反方向每次走一小步，再算梯度，再走，最终达到一个极小值点
# 强化学习中的作用: 计算梯度，从而更新神经网络或策略的参数

# requires_grad_()
# .backward() / .grad
a = torch.rand((1, 3), requires_grad=True)
func = (a[0, 0] * a[0, 1]) ** 2
func.backward()
print(a.grad)
# torch.no_grad() .detach() : TD 目标（参考答案）不带梯度，backward() 不会去更新 target 网络。因为 target 网络是“老师”，只提供参考答案，不该被学生的误差反向传播改掉
# 写法一：no_grad 包住
# with torch.no_grad():
#     td_target = rewards + 0.99 * q_target_net(next_states).max(1).values
# 写法二：detach 剪断
# td_target = rewards + 0.99 * q_target_net(next_states).max(1).values.detach()

# torch.nn.utils.clip_grad_norm_(params, max_norm)
# 是梯度裁剪工具，用来防止梯度爆炸、让训练更稳定。在 RL 里尤其常用，因为 RL 的梯度噪声比监督学习大得多，偶尔会出现异常大的梯度把网络“冲垮”
# backward() 算完梯度后，这个函数会检查所有参数梯度的整体范数（默认 L2 范数）。如果这个总范数超过了 max_norm，它就把所有梯度按同一个比例缩小，让总范数刚好等于 max_norm
