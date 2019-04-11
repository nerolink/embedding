from torch import nn
import torch.sparse
import javalang as jl
import javalang.tree as jlt
from model.DBN import DBN


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
    import pickle
    from GlobalVariable import GlobalVariable as gv

    dbn = DBN(layers=[1, 1, 1, 1], params=gv.dbn_params)
    with open('./dbn_obj', 'wb') as file_obj:
        pickle.dump(dbn, file_obj)

