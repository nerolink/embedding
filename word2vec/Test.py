from torch import nn
import torch.sparse
import javalang as jl
import javalang.tree as jlt


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=2,
            kernel_size=4,
            out_channels=4
        )

    def forward(self, x):
        self.conv.forward(x)


if __name__ == '__main__':
    ast_tree = jl.parse.parse(
        open('J:/sdp/projects/poi-2.0/src/java/org/apache/poi/hssf/record/Margin.java', 'rb').read())

