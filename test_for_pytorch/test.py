import torch
arr = torch.rand(3, 2, 2)
print(f'原张量: {arr}')
a = torch.split(arr, 3, dim=0)  # 传一个整数，表示每块大小，最后一块可能不足
print(f'第一种切法: {a}', len(a))
a = torch.split(arr, [1, 2], dim=0)  # 传一个列表，精确表示要分成的大小，列表元素和要等于该维度长度
print(f'第二种切法: {a}', len(a))
