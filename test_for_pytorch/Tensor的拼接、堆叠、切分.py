import torch
# cat stack split chunk 这四个Tensor操作主要服务于经验数据的批处理、轨迹管理以及内存优化
# torch.cat 与 torch.stack 在 RL 中最常见的用途是:
# 将从回放池（Replay Buffer）中采样的零散经验（Transitions）组装成可供神经网络处理的批量张量
# torch.chunk 与 torch.split 在 RL 中主要解决两个痛点: 处理变长轨迹和缓解显存压力

# 1. torch.cat(tensors, dim)  # 拼接
# 沿已有维度拼接，要求其他维度相同，返回维度数不变
a = torch.cat((torch.ones(1, 2, 3), torch.zeros(2, 2, 3)), dim=0)
print(a)

# 2. torch.stack(tensors, dim)  # 堆叠
# 在新维度上堆叠Tensor，所有输入张量形状必须完全相同
a = torch.stack([torch.ones(2, 3), torch.zeros(2, 3)], dim=0)
print(a)
# 如果不是在第0维上堆叠，可以在想象在要堆叠的维度上先 unsqueeze一个为1的维度，如何在这个维度上进行堆叠
# 这样你就能理解下面的这个操作了
a = torch.stack([torch.ones(2, 3), torch.zeros(2, 3)], dim=1)
print(a)

# 3. split、chunk
# 都是沿某个维度把Tensor切成多块，返回的是Tensor列表，区别在于: split 按"每块多大"切，chunk 按"切成几块"切
# split
arr = torch.randperm(10)
a = torch.split(arr, 3, dim=0)  # 传一个整数，表示每块大小，最后一块可能不足
print(a)
a = torch.split(arr, [2, 3, 5], dim=0)  # 传一个列表，精确表示要分成的大小，列表元素和要等于该维度长度
print(a)
# 多维也是一样
arr = torch.rand(3, 2, 2)
print(f'原张量: {arr}')
a = torch.split(arr, 3, dim=0)  # 传一个整数，表示每块大小，最后一块可能不足
print(f'第一种切法: {a}', len(a))
a = torch.split(arr, [1, 2], dim=0)  # 传一个列表，精确表示要分成的大小，列表元素和要等于该维度长度
print(f'第二种切法: {a}', len(a))

# chunk
# 语法基本和split一样，只不过传入的数字表示想要切分的块数，系统会尽量均分，实际块数可能达不到要求

