import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import os
import javalang as jl
import javalang.tree as jlt

features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam',
            'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']
head_names = {'org', 'bsh', 'com', 'javax', 'gnu', 'fr'}
types = [jlt.FormalParameter, jlt.BasicType, jlt.PackageDeclaration, jlt.InterfaceDeclaration,
         jlt.CatchClauseParameter, jlt.ClassDeclaration, jlt.MemberReference, jlt.SuperMemberReference,
         jlt.ConstructorDeclaration, jlt.ReferenceType, jlt.MethodDeclaration, jlt.VariableDeclarator, jlt.IfStatement,
         jlt.WhileStatement, jlt.DoStatement, jlt.ForStatement, jlt.AssertStatement, jlt.BreakStatement,
         jlt.ContinueStatement, jlt.ReturnStatement, jlt.ThrowStatement, jlt.SynchronizedStatement, jlt.TryStatement,
         jlt.SwitchStatement, jlt.BlockStatement, jlt.StatementExpression, jlt.TryResource, jlt.CatchClause,
         jlt.CatchClauseParameter, jlt.SwitchStatementCase, jlt.ForControl, jlt.EnhancedForControl]


class ProjectData(object):

    def __init__(self, data, vec_len, vocabulary_size):
        """
        :param data:[project_data,label]
        """
        self.data = data
        self.vec_len = vec_len
        self.vocabulary_size = vocabulary_size

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
        return self.vec_len

    def get_vocabulary_size(self):
        return self.vocabulary_size


def get_md_data(datas):
    """
    csv 文件对应的是类名
    :param datas: [[源码地址，csv文件地址],..,[目标源码地址,目标csv文件地址]]
    :return:[ProjectData,ProjectData]
    """
    word_dict = {}
    result = []
    max_length = 0
    for entry in datas:
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
                        print('file %s parse error!' % os.path.join(root, file))
                    finally:
                        file_obj.close()
        project_features = []
        labels = []
        for class_name, ast_vec in c2v.items():
            hand_craft = c2h[class_name]
            project_features.append([ast_vec, hand_craft, class_name])
            labels.append(c2l[class_name])
        ros = RandomOverSampler()
        project_features, labels = ros.fit_resample(project_features, labels)
        result.append([project_features.tolist(), labels.tolist()])
    for project_data in result:
        project_features = project_data[0]
        for feature in project_features:
            feature[0] = padding(feature[0], max_length)
    return [ProjectData(x, max_length, len(word_dict)) for x in result]


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
    hand_craft_data = (hand_craft_data[features].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))) \
        .values.tolist()  # 归一化
    return dict(zip(class_names, labels)), dict(zip(class_names, hand_craft_data))


def get_full_class_name(root, file):
    """
    返回一个java文件对应的完整类名
    :param root: 文件的路径
    :param file:遍历的文件
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
    if isinstance(_node, jlt.MethodInvocation) or isinstance(_node, jlt.SuperMethodInvocation):
        return str(_node.member) + "()"
    if isinstance(_node, jlt.ClassCreator):
        return str(_node.type.name)
    if type(_node) in types:
        return _node.__class__.__name__
    return None


def padding(ast_vec, target_len):
    """
    给ast向量填充0
    :param target_len:
    :param ast_vec:
    :return:
    """
    now_len = len(ast_vec)
    for i in range(0, target_len - now_len):
        ast_vec.append(0)
    return ast_vec


if __name__ == '__main__':
    csvs_path = "J:\\sdp\\csvs\\"
    projects_path = "J:\\sdp\\projects\\"

    # test_case = [[projects_path + 'camel-1.2', csvs_path + 'camel-1.2'],
    #              [projects_path + 'camel-1.4', csvs_path + 'camel-1.4'],
    #              [projects_path + 'camel-1.6', csvs_path + 'camel-1.6']]
    test_case = [[projects_path + 'xerces-1.4.4', csvs_path + 'xerces-1.4.4']]
    results = get_md_data(test_case)
    print(results[0].get_vocabulary_size())
    print(results[0].get_vec_len())
    print(len(results[0].get_ast_vectors()))
    print(len(results[0].get_ast_vectors()[1]))
