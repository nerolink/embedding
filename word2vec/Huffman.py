import os
from queue import PriorityQueue as PQueue
import pickle
import javalang as jl
from common.GlobalVariable import instance as global_var
from common.GlobalVariable import global_lock
from common.utils import encode_path, get_node_name, get_all_need_class_name, get_full_class_name
from common.CommonClass import *
import logging

logging.basicConfig(format=global_var.config['logging_format'], level=global_var.config['logging_level'])


def build_huffman_tree(project_name, gv=global_var):
    """
    构建霍夫曼树
    :param gv:
    :param project_name:
    :return:
    """
    global_lock[project_name].acquire()
    try:
        if gv.load_hf_tree(project_name):
            return gv.hf_root
        word_count = {}
        class_names = get_all_need_class_name(project_name)
        ## 统计此项目的word count
        for root, dirs, files in os.walk(gv.projects_source_dir):
            if not root.__contains__(project_name):
                continue
            for file in files:
                full_class_name = get_full_class_name(root, file)
                if full_class_name is not None and full_class_name in class_names:
                    with open(os.path.join(root, file), "r") as _file_obj:
                        try:
                            content = _file_obj.read()
                            if content is None:
                                continue
                            logging.debug('building huffman tree,processing file %s  ' % os.path.join(root, file))
                            ast_tree = jl.parse.parse(content)
                            for _path, _node in ast_tree:
                                name = get_node_name(_node)
                                if name is not None:
                                    word_count[name] = word_count.get(name, 0) + 1
                        except jl.parser.JavaSyntaxError:
                            logging.error('file %s syntax error!' % os.path.join(root, file))
                        except AttributeError:
                            logging.error('file %s attribute error' % os.path.join(root, file))
                        except UnicodeDecodeError:
                            logging.error('parse file %s unicode decode error' % os.path.join(root, file))
        pq = PQueue()
        for k, v in word_count.items():
            vi = VocabNode(k)
            vi.count = v
            pq.put(vi)
        del word_count
        while pq.qsize() > 1:
            nl = StemNode(pq.get(), pq.get())
            pq.put(nl)
        result = pq.get()
        # 确定huffman树的每一个节点的路径编码
        encode_path(result, [])
        gv.hf_root = result
        gv.dump_hf_tree(project_name, result)
        return result
    finally:
        global_lock[project_name].release()


# def get_node_name(_node):
#     if isinstance(_node, jlt.MethodInvocation) or isinstance(_node, jlt.SuperMethodInvocation):
#         return str(_node.member) + "()"
#     if isinstance(_node, jlt.ClassCreator):
#         return str(_node.type.name)
#     if isinstance(_node, jlt.Node):
#         return _node.__class__.__name__
#     # if type(_node) in gv.types:
#     #     return _node.__class__.__name__
#     return None











if __name__ == '__main__':
    pass
