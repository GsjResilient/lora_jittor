import torch
from functorch.dim import Tensor
import torch.nn.functional as F
from triton.language import dtype

A=torch.Tensor([[1,3],[4,5]])
B=torch.Tensor([[4],[3]])
delta_w = F.conv1d(
    A.unsqueeze(0),
    B.unsqueeze(-1),
    groups=2
).squeeze(0)
torch.set_printoptions(precision=4)
print(delta_w,delta_w.shape)
print('A un: ',A.unsqueeze(0).shape,'A: ',A.shape)
print('B un: ',B.unsqueeze(0).shape,'B: ',B.shape)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # 定义一个 nn.Conv1d 模块
# conv_module = nn.Conv1d(
#     in_channels=3,    # 输入通道数
#     out_channels=6,   # 输出通道数
#     kernel_size=2,    # 卷积核大小
#     bias=False
# )
#
# # 输入数据 (batch=1, channels=3, length=5)
# x = torch.randn(1, 3, 5)
#
# # 使用 nn.Conv1d 计算
# output_module = conv_module(x)
#
# # 使用 F.conv1d 实现相同操作
# weight = conv_module.weight  # 提取权重 (out_channels, in_channels, kernel_size)
# bias = conv_module.bias      # 提取偏置 (out_channels,)
# print(bias)
# output_function = F.conv1d(
#     input=x,
#     weight=weight,
#     groups=conv_module.groups
# )
#
# # 验证结果是否一致
# print(torch.allclose(output_module, output_function))  # 输出 True