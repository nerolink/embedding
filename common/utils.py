import os
from common.GlobalVariable import GlobalVariable as gv
from common.GlobalVariable import types, head_names, features
from common.CommonClass import VocabNode
import javalang.tree as jlt
import javalang as jl
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import logging

logging.basicConfig(format=gv.config['logging_format'], level=gv.config['logging_level'])


class ProjectData(object):
    """
    self.data=[project_data,label]
    project_data=[ast_vec, hand_craft, class_name]
    """

    def __init__(self, data, vec_len, vocabulary_size):
        """
        :param data:[project_data,label]
        """
        self.data = data
        self.vec_len = vec_len
        self.vocabulary_size = vocabulary_size
        self.data_size = len(data[1])

    def get_data_size(self):
        """
        :return: 数据集的大小
        """
        return self.data_size

    def get_labels(self):
        """
        :return: [label]
        """
        return self.data[1]

    def get_ast_vectors(self):
        """
        :return:[[ast_vec],[ast_vec]...]
        """
        return [x[0] for x in self.data[0]]

    def get_hand_craft_vectors(self):
        """
        :return:[[hand_craft_vec],[hand_craft_vec]...]
        """
        return [x[1] for x in self.data[0]]

    def get_class_names(self):
        """
        :return: [class_name,class_name......]
        """
        return [x[2] for x in self.data[0]]

    def get_vec_len(self):
        """
        :return: ast 向量需要填充到的长度
        """
        return self.vec_len

    def get_vocabulary_size(self):
        return self.vocabulary_size

    def generator_get_data_xy(self, batch_size):
        """
        keras 一直地u奥区
        返回({ast向量},{标签})的生成器
        :param batch_size:
        :return:np.array
        """
        length = len(self.data)
        count = 0
        while True:
            if count >= length:
                count = 0
            end = count + batch_size if count + batch_size <= length else length
            yield (np.array([padding_for_token(self.data[0][0][i], self.vec_len) for i in range(count, end)]), np.array(
                self.data[1][count:end]))
            count += batch_size

    def generator_get_data_x(self, batch_size):
        """
        返回({ast向量})的生成器
        :param batch_size:
        :return:np.array
        """
        length = len(self.data)
        count = 0
        while True:
            if count >= length:
                count = 0
            end = count + batch_size if count + batch_size <= length else length
            yield (np.array(
                [padding_for_token(self.data[0][0][i], self.vec_len) for i in range(count, end)]))
            count += batch_size

    def generator_get_data_xhy(self, batch_size):
        """
        返回({ast向量,手工标注向量},{标签})的生成器
        :param batch_size:
        :return:np.array
        """
        length = len(self.data)
        count = 0
        while True:
            if count >= length:
                count = 0
            end = count + batch_size if count + batch_size <= length else length
            yield (np.array([padding_for_token(self.data[0][0][i], self.vec_len) for i in range(count, end)]),
                   np.array(self.data[0][1][count:end]),
                   np.array(self.data[1][count:end]))
            count += batch_size


def get_md_data(datas, imbalance):
    """
    返回csv和源码文件对应的数据，具体哪些数据看ProjectData的接口
    :param imbalance:
    :param datas: [[源码地址，csv文件地址],..,[目标源码地址,目标csv文件地址]]
    :return:[ProjectData,ProjectData]
    """
    result = []
    word_dict = {}
    max_length = 0
    k = ''
    for i in range(len(datas)):
        entry = datas[i]
        k += (entry[0] + entry[1])
    cache = gv.load_cache(k)
    if cache is not None:
        [result, max_length, word_dict] = cache
        return [ProjectData(_x, max_length, len(word_dict)) for _x in result]
    for i in range(len(datas)):
        entry = datas[i]
        source_path = entry[0]
        csv_path = entry[1] if entry[1].endswith('.csv') else entry[1] + '.csv'
        # class name to label,class name to hand craft data
        c2l, c2h = extract_class_name_and_label(csv_path)
        # class name to ast_vec
        c2v = {}
        for root, dirs, files in os.walk(source_path):
            for file in files:
                full_class_name = get_full_class_name(root, file)
                if full_class_name is not None and full_class_name in c2l.keys():
                    file_obj = open(os.path.join(root, file), 'r')
                    try:
                        ast_tree = jl.parse.parse(file_obj.read())
                        ast_vec = []
                        for path, node in ast_tree:
                            node_name = get_node_name(node)
                            if node_name is not None:
                                if node_name not in word_dict.keys():
                                    word_dict[node_name] = len(word_dict)
                                ast_vec.append(word_dict[node_name])
                        max_length = max(max_length, len(ast_vec))
                        c2v[full_class_name] = ast_vec
                    except jl.parser.JavaSyntaxError:
                        logging.error('file %s syntax error!' % os.path.join(root, file))
                    except AttributeError:
                        logging.error('file %s attribute error' % os.path.join(root, file))
                    except UnicodeDecodeError:
                        logging.error('file %s unicode decode error' % os.path.join(root, file))
                    finally:
                        file_obj.close()
        project_features = []
        labels = []
        for class_name, ast_vec in c2v.items():
            hand_craft = c2h[class_name]
            project_features.append([ast_vec, hand_craft, class_name])
            labels.append(c2l[class_name])
        if i < len(datas) - 1:  # 对于最后一个数据集不做不平衡处理
            ros = imbalance
            project_features, labels = ros.fit_resample(project_features, labels)
            result.append([project_features.tolist(), labels.tolist()])
        else:
            result.append([project_features, labels])
    gv.dump_cache(k, [result, max_length, word_dict])
    return [ProjectData(_x, max_length, len(word_dict)) for _x in result]


def get_all_need_class_name(project_name):
    """
    返回csv文件里存在的类名
    :param project_name:
    :return:
    """
    import pandas as pd
    result = set()
    for root, dirs, files in os.walk(gv.csv_dir):
        for file in files:
            if not file.__contains__(project_name) or not file.endswith('csv'):
                continue
            df = pd.read_csv(os.path.join(root, file))
            class_name = df['file_name'].values.tolist()
            result.update(class_name)
    return result


def encode_path(_root, pre_path):
    """
    确定每一个huffman树叶子节点的路径，0表示左边，1表示右边
    :param _root:
    :param pre_path:
    :return:
    """
    if _root is None:
        return
    if type(_root) is VocabNode:
        _root.path = pre_path.copy()
        return
    for index, subNode in enumerate(_root.children):
        pre_path += [index]
        encode_path(subNode, pre_path)
        del pre_path[-1]


def get_full_class_name(root, file):
    """
    返回文件对应的类名
    :param root: 文件目录
    :param file: 文件名
    :return:
    """
    if not file.endswith('.java'):
        return None
    a_p = root.split(os.sep)
    package_name = ''
    find = False
    for i in range(len(a_p) - 1, -1, -1):
        package_name = a_p[i] + '.' + package_name
        if a_p[i] in head_names:
            find = True
            break
    return package_name + file[0:-5] if find else None


def get_node_name(_node):
    """
    给定javalang节点，返回符合要求的节点名称，如果不符合返回None
    :param _node: javalang 节点
    :return:
    """
    if isinstance(_node, jlt.MethodInvocation) or isinstance(_node, jlt.SuperMethodInvocation):
        return str(_node.member) + "()"
    if isinstance(_node, jlt.ClassCreator):
        return str(_node.type.name)
    if type(_node) in types:
        return _node.__class__.__name__
    return None


def watch_data(project_name):
    """
    用于查看用pickle保存的变量，如ast向量的长度-token_vec_length
    :param project_name:
    :return:
    """
    import pickle
    token_vec_len_path = os.path.join(gv.token_len_path, '%s.tvl' % project_name)
    hf_tree_path = os.path.join(gv.hf_tree_path, '%s.ht' % project_name)
    if os.path.exists(token_vec_len_path):
        with open(token_vec_len_path, 'rb') as file_obj:
            print(pickle.load(file_obj))
    if os.path.exists(hf_tree_path):
        with open(hf_tree_path, 'rb') as file_obj:
            print(pickle.load(file_obj))


def batch_getter(batch_size, *data):
    """
    给定集合生成batch
    :param batch_size:
    :param data: 任意数量的list或者np.array
    :return:
    """
    length = len(data[0])
    count = 0
    while count < length:
        last_count = count
        count = count + batch_size if (count + batch_size) <= length else length
        yield (d[last_count:count] if count < length else d[last_count:] for d in data)


def extract_class_name_and_label(csv_path):
    """
    输入csv文件的路径，np.array([完整类名]),np.array([对应的标签])
    :param csv_path csv文件的路径
    :return:{类名:[[hand_craft],[label]]}
    """
    hand_craft_data = pd.read_csv(csv_path)
    hand_craft_data.fillna(0)
    class_names = hand_craft_data['file_name'].values.tolist()
    labels = np.array(hand_craft_data['bug'].values.tolist())
    labels = [1 if x > 1 else x for x in labels]
    hand_craft_data = (hand_craft_data[features].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))).fillna(0)) \
        .values.tolist()  # 归一化
    return dict(zip(class_names, labels)), dict(zip(class_names, hand_craft_data))


def padding_for_token(ast_vec, target_len):
    """
    给ast字符向量填充0
    :param target_len:
    :param ast_vec:[1,3,5,4,1,5,.....]
    :return:[1,3,5,4,1,5,..0,0,0,0,....]
    """
    now_len = len(ast_vec)
    for i in range(0, target_len - now_len):
        ast_vec.append(0)
    return ast_vec


def padding_for_token_batch(ast_vecs, target_len):
    """
    给ast字符向量填充0
    :param target_len:
    :param ast_vecs:[[1,3,5,4,1,5,.....],....]
    :return:[[1,3,5,4,1,5,..0,0,0,0,....]....]
    """
    result = []
    for i in range(len(ast_vecs)):
        ast_vec = ast_vecs[i]
        now_len = len(ast_vec)
        for j in range(0, target_len - now_len):
            ast_vec.append(0)
        result.append(ast_vec)
    return result


def extract_hand_craft_file_name_with_label(train_path, test_path):
    """
    输入csv文件的相对路径，返回文件名列表
    :param train_path:  手工标注的训练集csv文件路径
    :param test_path:   手工标注的测试集csv文件路径
    :return:
    """
    hand_craft_train = pd.read_csv(train_path)
    hand_craft_train.fillna(0)
    hand_craft_train_file_name = hand_craft_train['file_name'].values.tolist()
    hand_craft_train_label = hand_craft_train['bug'].values.tolist()
    hand_craft_train_label = [1 if x > 1 else x for x in hand_craft_train_label]
    hand_craft_train_hand = hand_craft_train[features] \
        .apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) \
        .fillna(0).values.tolist()

    hand_craft_test = pd.read_csv(test_path)
    hand_craft_test.fillna(0)
    hand_craft_test_file_name = hand_craft_test['file_name'].values.tolist()
    hand_craft_test_label = hand_craft_test['bug'].values.tolist()
    hand_craft_test_label = [1 if x > 1 else x for x in hand_craft_test_label]
    hand_craft_test_hand = hand_craft_test[features] \
        .apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) \
        .fillna(0).values.tolist()

    return hand_craft_train_file_name, hand_craft_test_file_name, hand_craft_train_label, \
           hand_craft_test_label, hand_craft_train_hand, hand_craft_test_hand


def print_result(y_true, y_pred, model, sheet_name, project_name, train_name, test_name, dict_param):
    """ 打印并持久化结果到data/result/sheet_name.csv中
    :param y_true:
    :param y_pred:
    :param model: 模型的名称
    :param sheet_name: 结果的表名
    :param project_name:
    :param train_name:
    :param test_name:
    :param dict_param: 存储到表中的参数
    :return:
    """
    from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
    f_1 = f1_score(y_true=y_true, y_pred=[0 if n < gv.round_threshold else 1 for n in y_pred])
    mcc = matthews_corrcoef(y_true=y_true, y_pred=[-1 if n < gv.round_threshold else 1 for n in y_pred])
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    print('from model %s:' % model)
    print('\tf1-score:' + str(f_1))
    print('\tmcc:' + str(mcc))
    print('\tauc:' + str(auc))
    gv.persistence(dict_params=dict_param, project_name=project_name, train_name=train_name, test_name=test_name,
                   f_1=f_1, mcc=mcc, auc=auc, model=model, y_true=y_true, y_pred=y_pred, sheet_name=sheet_name)


if __name__ == '__main__':
    import numpy as np

    a = np.array([1, 1, 1, 1, 11, 1, 154, 8, 5, 7])
    b = np.array([8, 9, 4, 2, 7, 6, 8, 36, 4, 2])
    for (x, y) in batch_getter(3, a, b):
        print(x)
        print(y)
        print()
