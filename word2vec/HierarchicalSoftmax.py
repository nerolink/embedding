import torch.nn
import torch
from word2vec import Huffman as hf
import javalang as jl
import time
import sys
import pickle
import os
from common.utils import get_full_class_name, get_all_need_class_name
from common.GlobalVariable import GlobalVariable as gv
import logging

logging.basicConfig(format=gv.config['logging_format'], level=gv.config['logging_level'])


def init(hf_tree):
    vec_size = gv.w2v_cnn_params['vec_size']
    if hf_tree is None:
        return
    if type(hf_tree) is hf.VocabNode:
        gv.word_to_vec[hf_tree.word] = (torch.rand(vec_size, device=gv.device, dtype=gv.d_type) - 0.5) / vec_size
        gv.word_to_vec[hf_tree.word].requires_grad = gv.requires_grad
        gv.word_to_node[hf_tree.word] = hf_tree
    else:
        hf_tree.theta = (torch.rand(vec_size, device=gv.device, dtype=gv.d_type) - 0.5) / vec_size
        hf_tree.theta.requires_grad = gv.requires_grad
        init(hf_tree.children[0])
        init(hf_tree.children[1])


def forward_and_back(context, word):
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
    parent = gv.hf_root
    log_likelihood = 0
    for i in vocab_node.path:
        if i == 0:
            log_likelihood += torch.log(torch.sigmoid(parent.theta.dot(xw)))  # 太大了 TODO
        else:
            t = parent.theta.dot(xw)
            tmp = torch.sigmoid(t)
            log_likelihood += torch.log(1 - tmp)
        parent = parent.children[i]
    log_likelihood.backward()
    parent = gv.hf_root
    for i in vocab_node.path:
        if torch.isnan(parent.theta.grad).sum().cpu().numpy() > 0:
            logging.error(
                '%s is nan:%s,log_likelihood:%f,count:%d' % ('parent.theta.grad.data', word, log_likelihood, count))
            sys.exit(-1)
        parent.theta.data += gv.w2v_cnn_params['learning_rate'] * parent.theta.grad.data
        parent = parent.children[i]
    for w in context:
        if torch.isnan(xw.grad).sum().cpu().numpy() > 0:
            logging.error("xw.grad is nan:" + word + ':' + str(gv.count))
            sys.exit(-1)
        gv.word_to_vec[w].data += (gv.w2v_cnn_params['learning_rate'] / len(context)) * xw.grad.data


def forward_and_back_with_formula(context, word):
    if context is None or len(context) == 0:
        return
    word_to_vec = gv.word_to_vec
    xw = torch.zeros_like(word_to_vec[context[0]], dtype=gv.d_type)
    for w in context:
        xw.data += gv.word_to_vec[w].data
    xw.data = xw.data / len(context)
    vocab_node = gv.word_to_node[word]
    parent = gv.hf_root
    l_xw = torch.zeros_like(xw)
    for i in vocab_node.path:
        if torch.isnan(parent.theta).sum().cpu().numpy() > 0:
            logging.error('%s is nan:%s,count:%d' % ('parent.theta.grad.data', word))
            sys.exit(-1)
        l_xw += (1 - i - torch.sigmoid(parent.theta.dot(xw))) * parent.theta
        parent.theta += gv.w2v_cnn_params['learning_rate'] * (1 - i - torch.sigmoid(parent.theta.dot(xw))) * xw
        parent = parent.children[i]
    for w in context:
        if torch.isnan(xw).sum().cpu().numpy() > 0:
            logging.error("xw.grad is nan:" + word + ':' + str(gv.count))
            sys.exit(-1)
        gv.word_to_vec[w] += (gv.w2v_cnn_params['learning_rate'] / len(context)) * l_xw


def train(project_name):
    """
    使用项目的所有版本训练word to vec
    :param project_name:
    :return:
    """
    # TODO 要判断两个路径都存在
    if gv.load_word2vec(project_name) and gv.load_token_vec_length(project_name):
        return
    start = time.time()
    logging.warning('start build huffman')
    hf.build_huffman_tree(project_name)
    logging.warning('start init')
    init(gv.hf_root)
    logging.warning('start walk')
    file_count = 0
    class_names = get_all_need_class_name(project_name)
    for root, dirs, files in os.walk(gv.projects_source_dir):
        if not root.__contains__(project_name):
            continue
        for file in files:
            full_class_name = get_full_class_name(root, file)
            if full_class_name is not None and full_class_name in class_names:
                if gv.isDebug:
                    logging.debug('walk file ' + file)
                start_time = time.time()
                file_count += 1
                node_count = 0
                try:
                    logging.debug("processing file %s in hierarchical softmax" % os.path.join(root, file))
                    ast_tree = jl.parse.parse(open(os.path.join(root, file), 'rb').read())
                    for path, node in ast_tree:
                        name = hf.get_node_name(node)
                        if name is not None:
                            node_count += 1
                            context = hf.get_context(path, node)
                            forward_and_back_with_formula(context if context[0] is not None else context[1:], name)
                    gv.w2v_cnn_params['token_vec_length'] = max(gv.w2v_cnn_params.get('token_vec_length', 0),
                                                                node_count)
                except jl.parser.JavaSyntaxError:
                    logging.error('file %s syntax error' % os.path.join(root, file))
                except AttributeError:
                    logging.error('file %s attribute error' % os.path.join(root, file))
                except UnicodeDecodeError:
                    logging.error('parse file %s unicode decode error' % os.path.join(root, file))
                finally:
                    logging.info('     consume time %f seconds' % (time.time() - start_time))
                    logging.info('     total walk %d files' % file_count)
    gv.dump_word2vec_txt(project_name)
    gv.dump_word2vec(project_name)
    gv.dump_token_vec_length(project_name)
    logging.info('finish in %f minutes' % ((time.time() - start) / 60))


if __name__ == '__main__':
    with open(gv.data_path + 'token_vec_length_camel.data', 'rb') as file_obj:
        print(pickle.load(file_obj))
