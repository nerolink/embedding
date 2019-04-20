import torch.nn
import torch
from word2vec import Huffman as hf
import javalang as jl
import time
import sys
import pickle
import os
from common.utils import get_full_class_name, get_all_need_class_name,get_context
from common.GlobalVariable import instance as global_var
import logging
import threading

logging.basicConfig(format=global_var.config['logging_format'], level=global_var.config['logging_level'])
lock = threading.Lock()


def init(hf_tree, gv=global_var):
    if hf_tree is None:
        return
    if type(hf_tree) is hf.VocabNode:
        gv.word_to_vec[hf_tree.word] = (torch.rand(gv.w2v_cnn_params['vec_size'], device=gv.device,
                                                   dtype=gv.d_type) - 0.5) / gv.w2v_cnn_params['vec_size']
        gv.word_to_vec[hf_tree.word].requires_grad = gv.requires_grad
        gv.word_to_node[hf_tree.word] = hf_tree
    else:
        hf_tree.theta = (torch.rand(gv.w2v_cnn_params['vec_size'], device=gv.device, dtype=gv.d_type) - 0.5) / \
                        gv.w2v_cnn_params['vec_size']
        hf_tree.theta.requires_grad = gv.requires_grad
        init(hf_tree.children[0], gv)
        init(hf_tree.children[1], gv)


def forward_and_back(context, word, gv=global_var):
    gv.count += 1
    count = gv.count
    if len(context) == 0:
        return
    xw = torch.zeros_like(gv.word_to_vec[context[0]], dtype=gv.d_type)
    xw.requires_grad = gv.requires_grad
    for w in context:
        xw.data += gv.word_to_vec[w].data
    xw.data = xw.data / len(context)
    vocab_node = gv.word_to_node[word]
    current_node = gv.hf_root
    tmp = torch.tensor(data=0, requires_grad=True, dtype=torch.float64)
    log_likelihood = tmp.clone()
    # for i in vocab_node.path:
    #     if i == 0:
    #         log_likelihood += torch.log(torch.sigmoid(parent.theta.dot(xw)))  # 太大了 TODO
    #     else:
    #         t = parent.theta.dot(xw)
    #         tmp = torch.sigmoid(t)
    #         log_likelihood += torch.log(1 - tmp)
    #     parent = parent.children[i]

    """
    前向计算
    """
    for i in vocab_node.path:
        log_likelihood += (1 - i) * torch.log(torch.sigmoid(current_node.theta.dot(xw))) + i * torch.log(
            1 - torch.sigmoid(current_node.theta.dot(xw)))
        current_node = current_node.children[i]

    """
    反向求梯度
    """
    log_likelihood.backward()
    parent = gv.hf_root
    for i in vocab_node.path:
        if torch.isnan(parent.theta.grad).sum().cpu().numpy() > 0:
            logging.error(
                '%s is nan:%s,log_likelihood:%f,count:%d' % ('parent.theta.grad.data', word, log_likelihood, count))
        parent.theta.data += gv.w2v_cnn_params['learning_rate'] * parent.theta.grad.data
        parent = parent.children[i]
    for w in context:
        if torch.isnan(xw.grad).sum().cpu().numpy() > 0:
            logging.error("xw.grad is nan:" + word + ':' + str(gv.count))
        gv.word_to_vec[w].data += (gv.w2v_cnn_params['learning_rate'] / len(context)) * xw.grad.data


def forward_and_back_with_formula(context, word, gv=global_var):
    if context is None or len(context) == 0:
        return
    xw = torch.zeros_like(gv.word_to_vec[context[0]], dtype=gv.d_type)
    for w in context:
        xw.data += gv.word_to_vec[w].data
    xw.data = xw.data / len(context)
    vocab_node = gv.word_to_node[word]
    parent = gv.hf_root
    e = torch.zeros_like(xw)
    for i in vocab_node.path:
        if torch.isnan(parent.theta).sum().cpu().numpy() > 0:
            logging.error('%s is nan:%s,count:%d' % ('parent.theta.grad.data', word))
        q = torch.sigmoid(parent.theta.dot(xw))
        g = gv.w2v_cnn_params['learning_rate'] * (1 - i - q)
        e += g * parent.theta
        parent.theta += g * xw
        parent = parent.children[i]
    for w in context:
        if torch.isnan(xw).sum().cpu().numpy() > 0:
            logging.error("xw.grad is nan:" + word + ':' + str(gv.count))
        gv.word_to_vec[w] += (gv.w2v_cnn_params['learning_rate'] / len(context)) * e


def train(project_name, gv=global_var):
    """
    使用项目的所有版本训练word to vec
    :param gv:
    :param project_name:
    :return:
    """

    lock.acquire()
    con = gv.load_word2vec(project_name) and gv.load_token_vec_length(project_name)
    lock.release()
    if con:
        return
    start = time.time()
    logging.warning('start build huffman')
    hf.build_huffman_tree(project_name, gv)
    logging.warning('start init')
    init(gv.hf_root, gv)
    logging.warning('start walk')
    class_names = get_all_need_class_name(project_name)
    gv.w2v_cnn_params['token_vec_length'] = 0
    file_count = 0
    syntax_error_count = 0
    attr_error_count = 0
    uni_error_count = 0
    for root, dirs, files in os.walk(gv.projects_source_dir):
        if not root.__contains__(project_name):
            continue
        for file in files:
            full_class_name = get_full_class_name(root, file)
            if full_class_name is not None and full_class_name in class_names:
                start_time = time.time()
                node_count = 0
                try:
                    logging.debug("processing file %s in hierarchical softmax" % os.path.join(root, file))
                    ast_tree = jl.parse.parse(open(os.path.join(root, file), 'r').read())
                    for path, node in ast_tree:
                        name = hf.get_node_name(node)
                        if name is not None:
                            node_count += 1
                            context = get_context(path, node)
                            forward_and_back_with_formula(context if context[0] is not None else context[1:], name, gv)
                            # forward_and_back(context if context[0] is not None else context[1:], name)
                    gv.w2v_cnn_params['token_vec_length'] = max(
                        gv.w2v_cnn_params.get('token_vec_length', 0),
                        node_count)
                    file_count = file_count + 1
                except jl.parser.JavaSyntaxError:
                    syntax_error_count += 1
                    logging.error('file %s syntax error' % os.path.join(root, file))
                except AttributeError:
                    attr_error_count += 1
                    logging.error('file %s attribute error' % os.path.join(root, file))
                except UnicodeDecodeError:
                    uni_error_count += 1
                    logging.error('parse file %s unicode decode error' % os.path.join(root, file))
                finally:
                    logging.error('     consume time %f seconds' % (time.time() - start_time))
                    logging.error(
                        '     total walk %d files,syntax error file %d,'
                        'attribute error file %d,unicode decode error file %d' % (
                            file_count, syntax_error_count, attr_error_count, uni_error_count))

    print(file_count)
    gv.dump_word2vec_txt(project_name)
    gv.dump_word2vec(project_name)
    gv.dump_token_vec_length(project_name)
    logging.info('finish in %f minutes' % ((time.time() - start) / 60))


if __name__ == '__main__':
    train('camel')
