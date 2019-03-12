import torch.nn
import torch
import Huffman as hf
import GlobalVariable as gv
import os
import javalang.tree as jlt
import javalang as jl
import time
import sys


def init(hf_tree):
    vec_size = gv.params['vec_size']
    if hf_tree is None:
        return
    if type(hf_tree) is hf.VocabNode:
        gv.word_to_vec[hf_tree.word] = (torch.rand(vec_size, device=gv.device, dtype=gv.d_type) - 0.5) / vec_size
        gv.word_to_vec[hf_tree.word].requires_grad = True
        gv.word_to_node[hf_tree.word] = hf_tree
    else:
        hf_tree.theta = (torch.rand(vec_size, device=gv.device, dtype=gv.d_type) - 0.5) / vec_size
        hf_tree.theta.requires_grad = True
        init(hf_tree.children[0])
        init(hf_tree.children[1])


def forward_and_back(context, word):
    gv.count += 1
    count = gv.count
    if len(context) == 0:
        return
    xw = torch.zeros_like(gv.word_to_vec[context[0]], dtype=gv.d_type)
    xw.requires_grad = True
    for w in context:
        xw.data += gv.word_to_vec[w].data
    xw.data = xw.data / len(context)
    vocab_node = gv.word_to_node[word]
    parent = gv.hf_root
    log_likelihood = 0
    if count == 14:
        print(count)
    for i in vocab_node.path:
        if i == 0:
            log_likelihood += torch.log(torch.sigmoid(parent.theta.dot(xw)))  # 太大了 TODO
        else:
            t = parent.theta.dot(xw)
            tmp = torch.sigmoid(t)
            log_likelihood += torch.log(1 - tmp)
            # log_likelihood += torch.log(1 - torch.sigmoid(parent.theta.dot(xw)))
        parent = parent.children[i]
    log_likelihood.backward()
    parent = gv.hf_root
    for i in vocab_node.path:
        if torch.isnan(parent.theta.grad).sum().cpu().numpy() > 0:
            print('%s is nan:%s,log_likelihood:%f,count:%d' % ('parent.theta.grad.data', word, log_likelihood, count))
            sys.exit(-1)
        parent.theta.data += gv.params['learning_rate'] * parent.theta.grad.data
        parent = parent.children[i]
    for w in context:
        if torch.isnan(xw.grad).sum().cpu().numpy() > 0:
            print("xw.grad is nan:" + word + ':' + str(gv.count))
            sys.exit(-1)
        gv.word_to_vec[w].data += (gv.params['learning_rate'] / len(context)) * xw.grad.data


def train(project_source):
    hf.build_huffman_tree(project_source)
    init(gv.hf_root)
    for root, dirs, files in os.walk(project_source):
        for file in files:
            if file.endswith('.java'):
                try:
                    ast_tree = jl.parse.parse(open(os.path.join(root, file), 'rb').read())
                    for path, node in ast_tree:
                        name = hf.get_node_name(node)
                        if name is not None:
                            context = hf.get_context(path, node)
                            # start = time.time()
                            forward_and_back(context if context[0] is not None else context[1:], name)
                            # print(name+str(gv.word_to_vec[name]))
                            # print("finish forward and back in " + str(time.time() - start))
                except jl.parser.JavaSyntaxError:
                    print(file)
    with open("result.txt", 'w') as file_obj:
        for k, v in gv.word_to_vec.items():
            file_obj.write(str(k) + ":" + str(v) + "\n")


if __name__ == '__main__':
    train(gv.projects_source_dir + 'camel-1.2')
