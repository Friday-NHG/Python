import torch
# 背景：你写了个 RL 环境，每步返回的观测是 channel-last 的 (H, W, C) = (4, 5, 3)（很多环境/库默认这种格式）
# 但神经网络要 channel-first 的 (C, H, W)，并且要带 batch 维。另外你还要从一批观测里按索引取子集、把异常值（比如 > 40）置零。
# 环境给的观测：channel-last (H,W,C)
obs = torch.arange(4 * 5 * 3, dtype=torch.float32).reshape(4, 5, 3)
# 任务：
# 1. 转成模型输入格式：把 obs 从 (H,W,C) 变成 (C,H,W)，再加 batch 维得到 (1,C,H,W)。要求打印最终 shape，并验证这一步是视图（data_ptr 不变）
# 2. 验证"转置后不连续"：打印转置后张量的 stride() 和 is_contiguous()，并解释为什么 view 在这一步会失败（可以试着 view(-1) 看报错）
# 3. 取 batch 子集：假设你有 3 个观测拼成的 batch (3,C,H,W)（用 repeat 或 stack 造一个），然后按索引取出两个样本，打印结果 shape
# 4. 筛异常值：对第 3 步取出的 batch，用布尔索引把所有 > 40 的值置为 0，打印修改后的 batch
# 5. 变连续：把第 4 步的结果变成连续张量 z，打印 z.stride() 和 z.is_contiguous()

# """1"""
print(obs)
target_obs = obs.permute(2, 0, 1).unsqueeze(0)
print(target_obs.shape, obs.data_ptr(), target_obs.data_ptr())
# """2"""
print(obs.stride, target_obs.stride)
print(obs.is_contiguous(), target_obs.is_contiguous())
try:
    view1 = target_obs.view(-1, 12)
except Exception as e:
    print(e)
view2 = obs.view(-1, 12)
print(view2)
# """3"""
batch_3 = target_obs.repeat(3, 1, 1, 1)
sample = batch_3[0:2, ...]
print(sample.shape)
# """4"""
sample[sample > 40] = 0
print(sample)
# """5"""
z = sample.contiguous()
print(z.shape, z.is_contiguous())