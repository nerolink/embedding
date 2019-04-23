import os
from common.GlobalVariable import instance as gv
from common.GlobalVariable import types, head_names, features
from common.CommonClass import VocabNode
import javalang.tree as jlt
import javalang as jl
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import logging
from graphviz import Graph
from imblearn.over_sampling import RandomOverSampler

logging.basicConfig(format=gv.config['logging_format'], level=gv.config['logging_level'])
good_data = [['forrest-0.6', 'forrest-0.7'], ['poi-1.5', 'poi-2.0'], ['jedit-4.2', 'jedit-4.3'],
             ['velocity-1.4', 'velocity-1.5'], ['xalan-2.4', 'xalan-2.5'], ['xerces-1.2', 'xerces-1.3'],
             ['log4j-1.1', 'log4j-1.2']]


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


def get_md_data(datas, imbalance=RandomOverSampler()):
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


def get_node_name(node):
    """
    给定javalang节点，返回符合要求的节点名称，如果不符合返回None
    :param node: javalang 节点
    :return:
    """
    if isinstance(node, jlt.MethodInvocation) or isinstance(node, jlt.SuperMethodInvocation):
        return str(node.member) + "()"
    if isinstance(node, jlt.ClassCreator):
        return str(node.type.name)
    if type(node) in types:
        return node.__class__.__name__
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


def padding_for_vec_batch(batch_vec, desire_length):
    """
    给word2vec后的ast向量batch补0
    :param batch_vec:
    [
    [[...],[...]...],
    [[...],[...]...]
    ]
    :param desire_length:
    :return:
    """
    result = []
    padding_line = np.zeros_like(batch_vec[0][0]).tolist()
    for vec in batch_vec:
        diff = desire_length - len(vec)
        result.append(vec)
        for i in range(diff):
            result[-1].append(padding_line)
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


def print_result(y_true, y_pred, model, sheet_name, project_name, train_name, test_name, dict_params):
    """ 打印并持久化结果到data/result/sheet_name.csv中
    :param y_true:
    :param y_pred:
    :param model: 模型的名称
    :param sheet_name: 结果的表名
    :param project_name:
    :param train_name:
    :param test_name:
    :param dict_params: 存储到表中的参数
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
    gv.persistence(dict_params=dict_params, project_name=project_name, train_name=train_name, test_name=test_name,
                   f_1=f_1, mcc=mcc, auc=auc, model=model, y_true=y_true, y_pred=y_pred, sheet_name=sheet_name)


def z_score(data):
    """
    :param data: np.array([[...],[...],...])
    :return:
    """
    return (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)


def min_max_score(data):
    """
    :param data:  np.array([[...],[...],...])
    :return:
    """
    return (data - data.min(axis=1, keepdims=True)) / (
            data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True))


def correct_source(all_source):
    """
    纠正一些代码的错误
    :param all_source:所有项目的路径
    :return:
    """
    import threadpool
    from multiprocessing import cpu_count
    import time
    import re
    pool = threadpool.ThreadPool(cpu_count())
    enum_pattern = re.compile(r'[^a-zA-Z]enum')

    def get_syntax_error_file(project_path):
        """
        :param project_path: 特定一个项目的路径
        :return:
        """
        se_files = []
        ue_files = []
        ae_files = []
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if not file.endswith('.java'):
                    continue
                try:
                    jl.parse.parse(open(os.path.join(root, file), encoding='utf-8').read())
                except jl.parser.JavaSyntaxError:
                    se_files.append(os.path.join(root, file))
                except UnicodeDecodeError:
                    ue_files.append(os.path.join(root, file))
                except AttributeError:
                    ae_files.append(os.path.join(root, file))

        for file_path in se_files:
            cache = []
            for line in open(file_path, 'r').readlines():
                cache.append(line)
            with open(file_path, 'w') as file_obj:
                advance = False  # 表示遇到assert语句的时候，是否需要删除多行
                for line in cache:
                    if line.__contains__('assert'):
                        if line.__contains__(';'):
                            advance = False
                        else:
                            advance = True
                        continue
                    if advance:
                        if line.__contains__(';'):
                            advance = False
                        continue

                    enum_list = re.findall(enum_pattern, line)
                    for err in enum_list:
                        corr = err.replace('enum', 'enum_')
                        line = line.replace(err, corr)
                    line = line.replace('package ${packageName};', ' ')
                    file_obj.write(line if line.endswith('\n') else line + '\n')
        for file_path in se_files:
            print(file_path)
        for file_path in ue_files + ae_files:
            print(file_path)
        logging.error('finishing processing project:%s,processed %s files ' % (project_path, len(se_files)))

    list_var = []
    for pj_path in os.listdir(all_source):
        params = dict()
        params['project_path'] = os.path.join(all_source, pj_path)
        list_var.append((None, params))
    start = time.time()
    requests = threadpool.makeRequests(get_syntax_error_file, list_var)
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print('consume %d seconds' % (time.time() - start))


def get_children_list(node):
    """
    有的子节点是list，要再递归一层拿到子节点
    :param node: ast父节点
    :return: 返回这个节点的所有子节点
    """
    result = []
    for child in node.children:
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


def get_parent_name(path):
    """
    返回某个AST节点的父节点的名称
    :param path: AST节点对应的节点路径
    :return: 父节点名称
    """
    if path is None or len(path) == 0 or (len(path) == 1 and type(path[0]) == type(list)):
        return None
    last_index = -2 if type(path[-1]) is list else -1
    return get_node_name(path[last_index])


def get_context(path, node):
    """
    返回某个节点对应的父节点和子节点组成的上下文
    :param path:
    :param node:
    :return:
    """
    return [get_parent_name(path)] + get_children_list(node)


def parse_ast_tree(file_path):
    """
    解析AST并把错误封装起来
    :param file_path:
    :return:
    """
    ast_tree = None
    try:
        ast_tree = jl.parse.parse(open(file_path, 'r').read())
    except jl.parser.JavaSyntaxError:
        logging.error('parse file %s java syntax error' % file_path)
    except UnicodeDecodeError:
        logging.error('parse file %s unicode decode error' % file_path)
    except AttributeError:
        logging.error('parse file %s attribute error' % file_path)
    finally:
        return ast_tree


def draw_tree(node, dot, parent_num, child_num):
    """
    递归地画AST
    :param dot:
    :param node:
    :param parent_num:
    :param child_num:
    :return:
    """
    if node is None:
        return
    if type(node) == list:
        if len(node) == 0:
            return
        count = 0
        for child in node:
            draw_tree(child, dot, str(parent_num), str(child_num) + str(count))
            count = count + 1
    else:
        if isinstance(node, jlt.Node):
            node_name = get_node_name(node)
            if node_name is not None:
                node_num = str(parent_num) + str(child_num)
                dot.node(node_num, get_node_name(node), shape='box')
                dot.edge(str(parent_num), node_num)
                count = 0
                for child in node.children:
                    draw_tree(child, dot, node_num, str(count))
                    count = count + 1
            else:
                count = 0
                for child in node.children:
                    draw_tree(child, dot, str(parent_num), str(child_num) + str(count))
                    count = count + 1


def show_ast(file_path):
    """
    把AST画成矢量图
    :param file_path: str
    :return:
    """
    ast_tree = parse_ast_tree(file_path)
    if ast_tree is None:
        return
    dot = Graph(comment=file_path)
    draw_tree(ast_tree, dot, '', '0')
    pic_path = os.path.join(gv.data_path, 'picture')
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    dot.render(filename='ast', directory=pic_path, format='pdf', view=True)


def pre_process(df, metric, reverse, retain_col=None, group=None):
    if group is None:
        group = ['Model', 'train_project']
    if retain_col is None:
        retain_col = ['project_name']

    def process(t):
        if t[metric] < 0.5 and reverse:
            t[metric] = 1 - t[metric]
        return t

    retain_col.append(metric)
    return df.apply(func=process, axis=1).groupby(group).apply(
        lambda t: t[t[metric] == t[metric].max()].head(1)) \
        .sort_values(by=metric, ascending=False)[retain_col].reset_index()


def show_result(metric='auc', better=True, other_model_path=gv.plain_cnn_result_path, reverse=True):
    """
    :param reverse: 是否AUC取反
    :param metric:测量方法
    :param better: 是否只显示最好的
    :param other_model_path: 对比的模型的结果路径，gv.plain_cnn_result_path ，gv.dbn_result_path
    :return:
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    on_col = 'train_project'
    w2v = pre_process(pd.read_csv(gv.w2v_result_path), metric=metric, reverse=reverse, group=['train_project'])[[
        'train_project', metric]]
    pcnn = pre_process(pd.read_csv(other_model_path), metric=metric, reverse=reverse, group=['train_project'])[[
        'train_project', metric]]
    result = pd.merge(w2v, pcnn, on=on_col)
    if better:
        result = result[result['%s_x' % metric] + 0.04 > result['%s_y' % metric]]
    model_name = ''
    if other_model_path == gv.plain_cnn_result_path:
        model_name = 'plain'
    if other_model_path == gv.dbn_result_path:
        model_name = 'dbn'
    if other_model_path == gv.lr_result_path:
        model_name = 'lr'
    result.rename(columns={'%s_x' % metric: '%s_w2v' % metric, '%s_y' % metric: '%s_%s' % (metric, model_name)},
                  inplace=True)
    return result


def get_best(metric='auc', mode='w2v', reverse=False):
    """
    返回每一个模型，每一个项目里最好的
    :param metric:
    :param mode:
    :param reverse:对于auc<0.5的反转
    :return:
    """
    result_path = None
    retain_col = [metric]
    if mode == 'w2v':
        result_path = gv.w2v_result_path
        retain_col.append('vec_size')
    if mode == 'cnn':
        result_path = gv.plain_cnn_result_path
    if mode == 'dbn':
        result_path = gv.dbn_result_path
    return pre_process(pd.read_csv(result_path), metric=metric, reverse=reverse)


def draw_bar_chart(reverse=False, metric='auc'):
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set_style(style='darkgrid')
    r_w2v = pre_process(pd.read_csv(gv.w2v_result_path), metric, reverse, group=['train_project'])[[
        'train_project', metric]]
    r_dbn = pre_process(pd.read_csv(gv.dbn_result_path), metric, reverse, group=['train_project'])[[
        'train_project', metric]]
    r_cnn = pre_process(pd.read_csv(gv.plain_cnn_result_path), metric, reverse, group=['train_project'])[[
        'train_project', metric]]


def show_avg_files_and_buggy(train_file, test_file):
    if not train_file.endswith('.csv'):
        train_file = train_file + '.csv'
    if not test_file.endswith('.csv'):
        test_file = test_file + '.csv'
    df = pd.read_csv(os.path.join(gv.csv_dir, train_file))
    bug_count = len(df) - list(df.bug).count(0)
    # bug_count = df.bug.sum()
    file_count = len(df)
    df = pd.read_csv(os.path.join(gv.csv_dir, test_file))
    bug_count += len(df) - list(df.bug).count(0)
    # bug_count += df.bug.sum()
    file_count += len(df)
    return bug_count * 1.0 / file_count, file_count * 1.0 / 2


if __name__ == '__main__':
    # for pair in good_data:
    #     print(pair[0])
    #     print(show_avg_files_and_buggy(pair[0], pair[1]))
    show_ast('./klass.java')