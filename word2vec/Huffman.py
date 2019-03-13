import os
from queue import PriorityQueue as PQueue
import pickle
import javalang as jl
import javalang.tree as jlt

from word2vec import GlobalVariable as gv


class Node:
    def __init__(self, count=None, code=None, parent=None):
        self.count = count  # 用于组建霍夫曼树
        self.code = code  # 0 或 1 ,左0   右1
        self.parent = parent  # 父节点

    def __lt__(self, other):
        return self.count < other.count

    def __eq__(self, other):
        return self.count == other.count

    def __gt__(self, other):
        return self.count > other.count


class VocabNode(Node):
    def __init__(self, word):
        super(VocabNode, self).__init__()
        self.word = word
        self.path = None


class StemNode(Node):
    def __init__(self, left, right):
        """
        :param left: Node 类型，
        :param right: Node 类型 left.count<=right.count
        """
        super(StemNode, self).__init__(count=left.count + right.count)
        self.children = [left, right]
        left.parent = self
        right.parent = self
        left.code = 0
        right.code = 1
        self.theta = None


def build_huffman_tree(project_sources):
    if gv.isDebug and os.path.exists('./hf_tree.data'):
        with open('./hf_tree.data', 'rb') as file_obj:
            gv.hf_root = pickle.load(file_obj)
        return
    word_count = {}
    ## 统计此项目的word count
    for root, dirs, files in os.walk(project_sources):
        for file in files:
            if file.endswith(".java"):
                with open(os.path.join(root, file), "rb") as file_obj:
                    content = file_obj.read()
                    try:
                        ast_tree = jl.parse.parse(content)
                        for _path, _node in ast_tree:
                            name = get_node_name(_node)
                            if name is not None:
                                word_count[name] = word_count.get(name, 0) + 1
                    except jl.parser.JavaSyntaxError:
                        print(file)

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
    encode_path(result, [])
    gv.hf_root = result
    with open("hf_tree.data", 'wb') as file_obj:
        pickle.dump(result, file_obj)
    return result


def get_node_name(_node):
    if isinstance(_node, jlt.MethodInvocation) or isinstance(_node, jlt.SuperMethodInvocation):
        return str(_node.member) + "()"
    if isinstance(_node, jlt.ClassCreator):
        return str(_node.type.name)
    if isinstance(_node, jlt.Node):
        return _node.__class__.__name__
    # if type(_node) in gv.types:
    #     return _node.__class__.__name__
    return None


def get_parent_name(_path):
    if _path is None or len(_path) == 0 or (len(_path) == 1 and type(_path[0]) == type(list)):
        return None
    last_index = -2 if type(_path[-1]) is list else -1
    return get_node_name(_path[last_index])


def get_children_list(_node):
    result = []
    for child in _node.children:
        if type(child) is list:
            for _c in child:
                c_name = get_node_name(_c)
                if c_name is not None:
                    result.append(c_name)
        else:
            child_name = get_node_name(child)
            if child_name is not None:
                result.append(child_name)
    return result


def get_context(_path, _node):
    return [get_parent_name(_path)] + get_children_list(_node)


def extract_context_2_file(source_path, project_name):
    file_obj = open('./context_for_' + project_name + '.txt', 'wb')
    contexts = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            try:
                if file.endswith('.java'):
                    _tree = jl.parse.parse(open(os.path.join(root, file), 'rb').read())
                    for _path, _node in _tree:
                        node_name = get_node_name(_node)
                        if node_name is not None:
                            contexts.append([node_name] + get_context(_path, _node))
            except jl.parser.JavaSyntaxError:
                print('         error:' + file)
            finally:
                print(file)
    pickle.dump(contexts, file_obj)
    file_obj.close()


def encode_path(_root, pre_path):
    if _root is None:
        return
    if type(_root) is VocabNode:
        _root.path = pre_path.copy()
        return
    for index, subNode in enumerate(_root.children):
        pre_path += [index]
        encode_path(subNode, pre_path)
        del pre_path[-1]


# if __name__ == '__main__':
#     tree = jl.parse.parse(open("Collection.java").read())
#     count = 0
#     for path, node in tree:
#         s = '['
#         for p in path:
#             tmp = str(type(p)) if type(p) is not list else (str(type(p)) + ':' + str(len(p)))
#             s = s + tmp + ','
#         s += ']' + ',' + str(type(node))
#         print(s)
#         s = '       {' + str(type(node)) + '||'
#         for c in node.children:
#             s += str(type(c)) + ','
#         s += '}'
#         print(s)

if __name__ == '__main__':
    # extract_context_2_file(gv.projects_source_dir + 'camel', 'camel')
    file_obj = open('context_for_camel.txt', 'rb')
    context = pickle.load(file_obj)
    print(len(context))
