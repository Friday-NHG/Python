import torch
# Tensor(张量): 就是多维数组，是机器学习框架里表示数据的基本单位
# 机器学习里数据(样本)几乎都是多维的，任何机器学习任务，输入输出都可以看成多维数组
# Tenser的类型: 跟编程语言里的数据类型差不多

#  “”“Tensor的属性”“”
# Tensor的常用属性有 .shape .stride .dtype .device .layout .stride 等
# torch.shape : Tensor形状
a = torch.rand(2, 3)
print(a.shape)
# torch.stride : 每个维度往前走一步，在底层一维内存里跨多少个元素
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # stride = (3, 1)  意思是: 行+1 跳过3个元素, 列+1 跳过1个元素
# torch.dtype : Tensor类型
# torch.device : Tensor对象创建好后存储在哪个设备上(CPU\GPU)
# torch.layout : Tensor对象在内存中以何种方式存储(稀疏/稠密)
# 示例:
# a = torch.tensor([1, 2, 3], dtype=torch.float32, device=torch.device('cpu'))


# """Tensor的创建"""
# torch.tensor/ zeros / ones / rand / empty 等
# 1. torch.tensor
# 它的核心用法是从已有数据构造张量，并可在构造时通过关键字参数指定数据类型、设备等属性
# 它的基本签名是 torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)
# data可以接收列表（list）、元组（tuple）、NumPy的ndarray、标量（scalar）
# device: 指定设备，默认在 CPU 上构造
# 从列表创建Tensor
a = torch.tensor([[1, 2], [3, 4]])
print(a, type(a))
# 从元组创建Tensor
a = torch.tensor(((5, 6), (7, 8)))
print(a, type(a))
# 从标量构造零维张量
a = torch.tensor(3.14159)
print(a, type(a))
# 从空列表构造空张量
a = torch.tensor([])
print(a, type(a))

# 2. torch.ones(shape)  创建全一 tensor
a = torch.ones(2, 3)
print(a)

# 3. torch.eye(shape)  创建对角线为一 tensor
a = torch.eye(2, 2)
print(a)

# 4. torch.zeros(shape)  创建全零 tensor
a = torch.zeros(3, 4)
print(a)

# 5.创建和a相同形状的全零/全一 tensor
a = torch.zeros_like(a)
# a = torch.ones_like(3, 4)
print(a)

# 6. torch.rand(shape)  每个值来自均匀分布0—1
a = torch.rand(3, 4)
print(a)
a = torch.rand(3)  # 创建一个长度为3的向量
print(a)

# 7. torch.randint(start, end , shape)  生成指定区间内的随机整数张量
a = torch.randint(0, 11, (2, 2))
print(a)

# 8. torch.randn(shape)  生成服从标准正态分布的随机浮点数张量
a = torch.randn(3, 4)
print(a)
# torch.normal(mean, std, shape)  自定义高斯分布

# 9. torch.randperm(n)  返回一个一维张量，相当于把[0, 1, 2, ..., n-1] 这 n 个整数随机打乱
a = torch.randperm(10)
print(a)

# 10. torch.arange(start, end, step)  返回等差序列
a = torch.arange(0, 10, 2)  # 从0-10，步长为2
print(a)

# 11. torch.linspace(start, end, step)  在区间[start, end]内均匀生成steps个点，包含起始值和结束值。
a = torch.linspace(0, 10, 3)  # 得到从0-10 等间隔划分的3个数值
print(a)

# 12. torch.empty(shape)  创建一个指定形状的张量，但其中的数据是未初始化的，内存里原本是什么，张量里就是什么
a = torch.empty(2, 3)
print('12:', a)
# 可以和原地(in-place)填充方式配合
a = torch.empty(2, 3).uniform_(0, 2)  # 0~2均匀分布
print(a)
a = torch.empty(2, 3).normal_(1, 2)
print(a)

import numpy as np  # python的一个库，和torch.tensor差不多
a = np.array([[1, 2], [3, 4]])
print(a)


# """Tensor的常用运算: """
# 一.加减乘除
# 有四种方式运算加减法
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c1 = a + b
c2 = torch.add(a, b)
c3 = a.add(b)
a.add_(b)  # 不同于前三种，运算的结果会更新为相加的结果  ##此为原地操作，通常用_ 下划线来表示
# 减法、乘法(哈达玛积: 对应元素相乘)和除法 写法同加法，换成 （-号 和 sub） (*号 和 mul)  (/号 和 div)

# 二.幂运算
a = torch.rand(2, 2)
#
b = torch.pow(a, 2)
b = a.pow(2)
b = a**2
a.pow_(2)
#
b = torch.exp(a)
b = a.exp_()

# 三.开方运算
print(a.sqrt_())
print(a.sqrt_())

# 四.对数运算
print(torch.log2(a))
print(torch.log10(a))
print(torch.log(a))  # 不写就默认底数为e
print(torch.log_(a))


# 五.矩阵运算
# 1.二维矩阵乘法
a = torch.ones(2, 3)
b = torch.ones(3, 2)
#
c4 = a @ b
a.matmul(b)
c5 = torch.matmul(a, b)
# 上面三个功能完全一致，而且适合一维到高维，下面两个是针对二维的简化写法
a.mm(b)
c6 = torch.mm(a, b)
#
# 对于高维的Tensor(dim > 2) 定义其矩阵乘法仅在最后的两个维度上，前面的维度必须保持一致
# 并且运算操作只有 torch.matmul()
print(torch.ones(1, 2, 3, 4))
# 形状
# ┌─────────────────────────────────────────┐
# │  dim0 = 0  （唯一的 1 个 batch）          │
# │  ┌───────────────────────────────────┐  │
# │  │ dim1 = 0  ── 通道 0                │  │
# │  │   ┌─────────────────────────┐     │  │
# │  │   │ 1  1  1  1   ← 行 0     │     │  │
# │  │   │ 1  1  1  1   ← 行 1     │     │  │
# │  │   │ 1  1  1  1   ← 行 2     │     │  │
# │  │   └─────────────────────────┘     │  │
# │  │       3 行 × 4 列                  │  │
# │  ├───────────────────────────────────┤  │
# │  │ dim1 = 1  ── 通道 1                │  │
# │  │   ┌─────────────────────────┐     │  │
# │  │   │ 1  1  1  1   ← 行 0     │     │  │
# │  │   │ 1  1  1  1   ← 行 1     │     │  │
# │  │   │ 1  1  1  1   ← 行 2     │     │  │
# │  │   └─────────────────────────┘     │  │
# │  │       3 行 × 4 列                  │  │
# │  └───────────────────────────────────┘  │
# └─────────────────────────────────────────┘
