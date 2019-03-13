import os

import torch.nn as nn
import javalang.tree as jlt
import torch


class A:
    attrs = ('a', 'b', 'c')


class B(A):
    attrs = ('d', 'e',)


# if __name__ == '__main__':
#     xw = torch.rand(5, requires_grad=True)
#     p = 1
#     for i in range(0, 100):
#         t = torch.rand(5, requires_grad=True)
#         p *= torch.sigmoid(xw.dot(t))
#     p.backward()
#     print(xw.grad)

if __name__ == '__main__':
    count = 0
    for root, dirs, files in os.walk('R:\\camel'):
        for file in files:
            if file.endswith('.java'):
                count += 1
    print(count)
