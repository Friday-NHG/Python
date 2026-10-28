import torch
# """Pytorch中的广播机制"""
# PyTorch 的广播机制，简单说就是：当两个形状不同的Tensor进行运算时
# 如果满足特定规则，PyTorch 会自动“虚拟地”扩展其中较小的Tensor，使其形状与较大的张量匹配，从而完成运算，且不需要实际复制数据
# 规则:
# 从右往左看每一个维度，如果每一个都满足: 相等 / 其中一个是1
# 那么他们就可以广播
# 广播的本质是: 某个轴大小为 1 的Tensor，在这个轴上被"复制扩展"到与另一个Tensor相同大小
a = torch.rand(2, 2, 3)
print(a)
b = torch.rand(1, 1, 3)
print(b)
c = a + b
print(c, c.shape)
# 运算时，广播后的扩展，等同于把原有的进行复制，但并没有真正复制


# “”“常用形状变换操作”“”
# RL 中形状变换的根本驱动力是：环境交互产生的是“单步、无 batch、channel-last、可能变长”的数据
# 而神经网络训练需要“带 batch、channel-first、定长、特定维度顺序”的数据
# 形状变换在他们两者之间的桥梁

# 两个基本概念: 视图、拷贝
# 视图（view）：不复制数据，只是换一种"看法"去读同一块内存，改视图会改到原Tensor，因为它们是同一块数据
# 拷贝（copy）：把数据完完整整再复制一份，放到新的内存里。改副本不影响原件，但费内存、费时间

# PyTorch 中常用的形状变换操作，按“是否改变数据布局”和“是否复制数据”这两个维度来理解，会清晰很多
# 核心要点是：优先使用返回视图（view）的操作以避免不必要的拷贝，但要注意视图与原Tensor共享内存带来的副作用


# 一.改变形状，不改变维度顺序
# 这类操作只重新解释数据，不交换维度(view、reshape、flatten)
# 1. view
# 改变形状，返回视图，零拷贝(支持-1自推断)
a = torch.rand(2, 6)
print(a)
b = a.view(3, 4)
print(b)
# 限制: view 要求读取数据的逻辑顺序(按 shape 的索引从外到内遍历元素的顺序)和内存顺序(元素在内存地址中的实际排列顺序)相一致，俗称内存连续

# 2. reshape
# view升级版，能返回视图就返回视图，如果不连续，它会自动先拷贝一份变成连续的，再重塑
a = torch.rand(3, 8)
print(a)
b = a.reshape(12, 2)
print(b)
# -1 的用法：两者都支持用 -1 让 PyTorch 自己算这一维多大。
b = a.reshape(-1, 6)   # 告诉它"每行 6 个，行数你自己算"
print(b)

# 3. flatten
# flatten(start_dim, end_dim) 的作用是将张量中从 start_dim 到 end_dim 的连续维度“压平”成一个维度，而其他维度保持不变
# 它本质是调用了reshape，如果不指定参数，整个张量将被展平为一维
a = torch.rand(2, 3, 4)
print(a)
print(a.flatten(1, 2))  # 维度也是从0开始算起的


# 二.改变维度数量
# 这类操作增加或删除维度，但维度里的数据不变(squeeze、unsqueeze)
# 1. squeeze(挤压)
# 把大小为 1 的维度删掉，大小为 1 的维度本质上不装任何数据，只是起到一个表示作用
a = torch.rand(1, 4, 1)
print(a)
a.squeeze()          # 形状变成 (4,)，所有大小为 1 的维度都被删了
print(a.shape)
a.squeeze(0)         # 只删第 0 维，形状变成 (4, 1)
print(a.shape)

# 2. unsqueeze(反挤压)
# 在指定位置插入一个大小为 1 的维度
a = torch.rand(4)
print(a)
a.unsqueeze(0)       # 在第 0 维前插一个 1，形状变成 (1, 4)
print(a.shape)
a.unsqueeze(1)       # 在第 1 维位置插一个 1，形状变成 (4, 1)
print(a.shape)
# 和广播机制配合
a = torch.rand(3)
b = torch.zeros(3, 3)
print(b + a.unsqueeze(0))  # (1,3) 看起来就是沿行方向广播
print(b + a.unsqueeze(1))  # (3,1) 看起来就是沿列方向广播


# 三.交换与重排维度
# 这类操作真的改变维度顺序，比如把"通道"从最后挪到最前(transpose、permute)
# 1. transpose
# 交换两个维度
a = torch.rand(2, 3)
print(a)
print(a.transpose(0, 1))    # 交换第 0 和第 1 维，形状 (2,3) → (3,2)

# 2. permute
# 一次性重排所有维度，要手动给出完整的新顺序
a = torch.rand(2, 3, 5)
print(a)
print(a.permute(2, 0, 1))  # 这个的写法是在所有维度上写上换成哪一个维度，原第2维→第0位，原第0维→第1位，原第1维→第2位
# 这两个操作都返回视图，零拷贝，内存还是同一块内存，只是读法变了
# 但是重排后张量在内存里不再连续（数据还是原来那样排，只是你读的顺序变了），这时如果再用 view 就会报错


# 四.扩展与复制
# 这类操作把Tensor"变大"，但两种方式代价完全不同(expand、repeat)
# 1. expand（扩展）
# 只能把大小为 1 的维度扩展成更大。它不分配新内存，靠把该维步长设为 0 来实现——步长为 0 意味着"沿这维走一步，内存地址不动"，反复读同一份数据
a = torch.rand(1, 4)
a.expand(3, 4)       # 形状 (1,4) → (3,4)，零拷贝
# expand 的限制：因为多个逻辑位置共享同一内存，不能对结果做原地修改（比如 y[0,0] = 5 会出错或影响其他位置），否则会污染共享数据。

# 2. repeat（重复）
# 沿各维度真的复制数据，分配新内存
a = torch.rand(1, 4)
a.repeat(3, 1)  # 复制成(3, 4)，占3倍内存


# “”“索引”“”
# 常用索引操作
# 1. 基本查询方式()
a = torch.rand(2, 3, 5)
print(a)
print(a[1, 2, 4])  # 第二个表格，第三行，第五个数值
print(a[0, 1])  # 第一个表格，第二行，所有列
print(a[:, :, 0])  # 每个表格，每一行，第一列( : 也是包头不包尾的)
print(a[0, :, 1])  # 第一个表格，所有行，第二列
print(a[..., 0])  # ... 表示所有维全保留，在前面就是前面的全保留，在后面就是后面的全保留
# 此处表示所有表格，所有行的第一列
print(a[..., -1])  # -1 表示最后

# 2. 布尔索引(按条件查找)
a = torch.rand(2, 3)
print(a)
print(a[a > 0.5])  # tensor([条件]) 返回值是一维Tensor，里面装着所有符合条件的数值
a[a > 0.5] = 0  # 利用布尔索引去修改原Tensor里的数值
print(a)

# 3. gather / scatter
# gather(dim, index): 沿 dim 按 index 取值
# scatter(dim, index, src): 沿 dim 按 index 吧src中的值写入调用scatter的张量里去
# 去找它真正的用途写在这里。。。。

# 4. index_select(dim, index)
# 在某个维度上去按索引选一些数据
a = torch.rand(1, 3)
print(a)
print(a.index_select(1, torch.tensor([1, 1, 2])))
# 最实在的用途：按编号批量"点名取数据"
