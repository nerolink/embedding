from torch import nn
import torch.sparse
import javalang as jl
import javalang.tree as jlt
from model.DBN import DBN
from common.GlobalVariable import instance as global_var


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


# if __name__ == '__main__':
#     import javalang as jl
#     import os
#
#     se_files = []
#     file_count = 0
#     ud = 0
#     for root, dirs, files in os.walk(global_var.projects_source_dir):
#         for file in files:
#             if not file.endswith('.java'):
#                 continue
#             print('processing file %s' % os.path.join(root, file))
#             file_count += 1
#             try:
#                 jl.parse.parse(open(os.path.join(root, file), encoding='utf-8').read())
#             except jl.parser.JavaSyntaxError:
#                 se_files.append(os.path.join(root, file))
#             except UnicodeDecodeError:
#                 ud += 1
#             except AttributeError:
#                 ud += 1
#     with open('what', 'w') as file_obj:
#         for file_path in se_files:
#             file_obj.write(file_path + '\n')
#     # error_file = []
#     # for file_path in open('what', 'r').readlines():
#     #     if file_path.endswith('\n'):
#     #         file_path = file_path[0:-1]
#     #     try:
#     #         jl.parse.parse(open(file_path, 'r').read())
#     #     except jl.parser.JavaSyntaxError:
#     #         error_file.append(file_path)
#     # with open('what', 'w') as file_obj:
#     #     file_obj.write(file_path if file_path.endswith('\n') else file_path + '\n')
if __name__ == '__main__':
    import re

    case = 'Assertion.assert(el1.getParentNode() != null);Assertion.assert(el1.getParentNode() != null);'
    pattern = re.compile(r'$Assertion.assert\((.+)\);^')
    print(re.search(r'Assertion.assert\((.+)\)', case).group(1))
    print(re.sub(pattern, 'assert \1', case))
