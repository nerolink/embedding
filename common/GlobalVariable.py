import javalang.tree as jlt
import torch
import pickle
import os
import time
import numpy as np
import xlrd
from imblearn.over_sampling import RandomOverSampler
from openpyxl import load_workbook
import hashlib
import logging

types = [jlt.FormalParameter, jlt.BasicType, jlt.PackageDeclaration, jlt.InterfaceDeclaration,
         jlt.CatchClauseParameter, jlt.ClassDeclaration, jlt.MemberReference, jlt.SuperMemberReference,
         jlt.ConstructorDeclaration, jlt.ReferenceType, jlt.MethodDeclaration, jlt.VariableDeclarator,
         jlt.IfStatement, jlt.WhileStatement, jlt.DoStatement, jlt.ForStatement, jlt.AssertStatement,
         jlt.BreakStatement, jlt.ContinueStatement, jlt.ReturnStatement, jlt.ThrowStatement,
         jlt.SynchronizedStatement, jlt.TryStatement, jlt.SwitchStatement, jlt.BlockStatement,
         jlt.StatementExpression, jlt.TryResource, jlt.CatchClause, jlt.CatchClauseParameter,
         jlt.SwitchStatementCase, jlt.ForControl, jlt.EnhancedForControl]
features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa',
            'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']
head_names = {'org', 'bsh', 'com', 'javax', 'gnu', 'fr'}
sheet_names = ['cnn_w2c', 'cnn_plain', 'dbn']
# projects = {
#     'camel': ['camel-1.2', 'camel-1.4', 'camel-1.6'],
#     'forrest': ['forrest-0.6', 'forrest-0.7', 'forrest-0.8'],
#     'jedit': ['jedit-3.2.1', 'jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3'],
#     'ivy': ['ivy-1.1', 'ivy-1.4', 'ivy-2.0'],
#     'log4j': ['log4j-1.0', 'log4j-1.1', 'log4j-1.2'],
#     'lucene': ['lucene-2.0', 'lucene-2.2', 'lucene-2.4'],
#     'poi': ['poi-1.5', 'poi-2.0', 'poi-2.5.1', 'poi-3.0'],
#     'synapse': ['synapse-1.0', 'synapse-1.1', 'synapse-1.2'],
#     'velocity': ['velocity-1.4', 'velocity-1.5', 'velocity-1.6.1'],
#     'xalan': ['xalan-2.4', 'xalan-2.5', 'xalan-2.6', 'xalan-2.7'],
#     'xerces': ['xerces-1.2', 'xerces-1.3', 'xerces-1.4.4']
# }

projects = {
    # 'camel': ['camel-1.2', 'camel-1.4', 'camel-1.6'],
    # 'forrest': ['forrest-0.6', 'forrest-0.7', 'forrest-0.8'],
    # 'jedit': ['jedit-3.2.1', 'jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3'],
    # 'ivy': ['ivy-1.1', 'ivy-1.4', 'ivy-2.0'],
    # 'log4j': ['log4j-1.0', 'log4j-1.1', 'log4j-1.2'],
    # 'lucene': ['lucene-2.0', 'lucene-2.2', 'lucene-2.4'],
    # 'poi': ['poi-1.5', 'poi-2.0', 'poi-2.5.1', 'poi-3.0'],
    # 'synapse': ['synapse-1.0', 'synapse-1.1', 'synapse-1.2'],
    # 'velocity': ['velocity-1.4', 'velocity-1.5', 'velocity-1.6.1'],
    'xalan': ['xalan-2.4', 'xalan-2.5', 'xalan-2.6', 'xalan-2.7'],
    'xerces': ['xerces-1.2', 'xerces-1.3', 'xerces-1.4.4']
}
candidate = {'num_filter'}


class GlobalVariable:
    word_to_vec = {}
    word_to_node = {}
    d_type = torch.float64
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    # 用于四舍五入
    round_threshold = 0.5

    w2v_cnn_params = {'Model': 'cnn', 'vec_size': 50, 'learning_rate': 0.01, 'round_threshold': 0.5,
                      'token_vec_length': 0, 'batch_size': 32, 'filters': 50, 'kernel_size': 20, 'mcc': None,
                      'hand_craft_input_dim': 20, 'pool_size': 2, 'hidden_units': 150, 'epochs': 1, 'metrics': ['acc'],
                      'project_name': None, 'train_project': None, 'test_project': None, 'time_stamp': None,
                      'auc': None, 'f1-score': None, 'use_cuda': True}

    # w2v_cnn_params = {'Model': 'cnn', 'vec_size': 80, 'learning_rate': 0.01, 'round_threshold': 0.5,
    #                   'token_vec_length': 0, 'batch_size': 32, 'filters': 50, 'kernel_size': 20, 'mcc': None,
    #                   'hand_craft_input_dim': 20, 'pool_size': 2, 'hidden_units': 150, 'epochs': 10, 'metrics': ['acc'],
    #                   'project_name': None, 'train_project': None, 'test_project': None, 'time_stamp': None,
    #                   'auc': None, 'f1-score': None, 'use_cuda': True}

    plain_cnn_params = {'input_dim': 3709, 'output_dim': 30, 'input_length': 2405, 'filters': 50, 'kernel_size': 20,
                        'pool_size': 2, 'hidden_units': 150, 'hand_craft_input_dim': 20, 'metrics': ['acc'],
                        'batch_size': 32, 'epochs': 1, 'imbalance': RandomOverSampler(), 'regenerate': False}

    dbn_params = {'output_size': 100, 'hidden_layer_num': 3, 'epochs': 1, 'batch_size': 10,
                  'imbalance': RandomOverSampler(), 'learning_rate': 0.1, 'regenerate': False}

    config = {'logging_level': logging.WARNING,
              'logging_format': '%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
              'remake': False}
    metrics = ['acc']
    projects_source_dir = "J:\\sdp\\projects\\"
    csv_dir = "J:\\sdp\\csvs\\"
    hf_root = None
    training_data = []
    debug_map = {}
    count = 0
    isDebug = True
    requires_grad = False
    sigmoid_threshold = 6
    data_path = '../data/'
    data_cache = '../data/cache'
    hf_tree_path = data_path + 'hf_tree'
    token_len_path = data_path + 'token_vec_len'
    word_to_vec_path = data_path + 'word_to_vec'
    result_path = data_path + 'result'
    current_project = None

    @staticmethod
    def load_hf_tree(project_name):
        if project_name is None or GlobalVariable.config['remake']:
            return None
        cache_dir = os.path.join(GlobalVariable.hf_tree_path, '%s.ht' % project_name)
        if not os.path.exists(cache_dir):
            GlobalVariable.hf_root = None
            return False
        with open(cache_dir, 'rb') as file_obj:
            GlobalVariable.hf_root = pickle.load(file_obj)
        GlobalVariable.current_project = project_name
        return True

    @staticmethod
    def dump_hf_tree(project_name, hf_tree):
        if project_name is None:
            return None
        if not os.path.exists(GlobalVariable.hf_tree_path):
            os.mkdir(GlobalVariable.hf_tree_path)
        cache_dir = os.path.join(GlobalVariable.hf_tree_path, '%s.ht' % project_name)
        with open(cache_dir, 'wb') as file_obj:
            pickle.dump(hf_tree, file_obj)

    @staticmethod
    def load_token_vec_length(project_name):
        if project_name is None or GlobalVariable.config['remake']:
            return None
        tvl_path = os.path.join(GlobalVariable.token_len_path, '%s.tvl' % project_name)
        if not os.path.exists(tvl_path):
            GlobalVariable.w2v_cnn_params['token_vec_length'] = 0
            return False
        with open(tvl_path, 'rb') as file_obj:
            GlobalVariable.w2v_cnn_params['token_vec_length'] = pickle.load(file_obj)
        GlobalVariable.current_project = project_name
        return True

    @staticmethod
    def dump_token_vec_length(project_name):
        if project_name is None:
            return None
        if not os.path.exists(GlobalVariable.token_len_path):
            os.mkdir(GlobalVariable.token_len_path)
        tvl_path = os.path.join(GlobalVariable.token_len_path, '%s.tvl' % project_name)
        with open(tvl_path, 'wb') as file_obj:
            pickle.dump(GlobalVariable.w2v_cnn_params['token_vec_length'], file_obj)

    @staticmethod
    def load_word2vec(project_name):
        if project_name is None or GlobalVariable.config['remake']:
            return None
        w2v_path = os.path.join(GlobalVariable.word_to_vec_path,
                                '%s_%d.w2v' % (project_name, GlobalVariable.w2v_cnn_params['vec_size']))
        if not os.path.exists(w2v_path):
            GlobalVariable.word_to_vec = {}
            return False
        with open(w2v_path, 'rb') as file_obj:
            GlobalVariable.word_to_vec = pickle.load(file_obj)
        GlobalVariable.current_project = project_name
        return True

    @staticmethod
    def dump_word2vec(project_name):
        """
        只和vec_size有关
        :param project_name:
        :return:
        """
        if project_name is None:
            return
        if not os.path.exists(GlobalVariable.word_to_vec_path):
            os.mkdir(GlobalVariable.word_to_vec_path)
        w2v_path = os.path.join(GlobalVariable.word_to_vec_path,
                                '%s_%d.w2v' % (project_name, GlobalVariable.w2v_cnn_params['vec_size']))
        with open(w2v_path, 'wb') as file_obj:
            pickle.dump(GlobalVariable.word_to_vec, file_obj)

    @staticmethod
    def dump_word2vec_txt(project_name):
        if project_name is None:
            return
        if not os.path.exists(GlobalVariable.word_to_vec_path):
            os.mkdir(GlobalVariable.word_to_vec_path)
        w2v_path = os.path.join(GlobalVariable.word_to_vec_path,
                                '%s_%d.txt' % (project_name, GlobalVariable.w2v_cnn_params['vec_size']))
        with open(w2v_path, 'w') as file_obj:
            for k, v in GlobalVariable.word_to_vec.items():
                file_obj.write(str(k) + ":" + str(list(v.data.cpu().numpy())) + "\n")
        return

    @staticmethod
    def dump_cache(file_name, obj):
        if file_name is None:
            return
        m2 = hashlib.md5()
        file_name = m2.update(file_name.encode('utf-8'))
        cache_path = os.path.join(GlobalVariable.data_cache, file_name)
        with open(cache_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    @staticmethod
    def load_cache(file_name):
        if file_name is None or GlobalVariable.config['remake']:
            return None
        m2 = hashlib.md5()
        file_name = m2.update(file_name.encode('utf-8'))
        cache_path = os.path.join(GlobalVariable.data_cache, file_name)
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    @staticmethod
    def persistence(dict_params, project_name, train_name, test_name, f_1, mcc, auc, model, y_true, y_pred,
                    sheet_name=None):
        import pandas as pd
        print('begin persistence')
        gv = GlobalVariable
        dict_params = dict_params.copy()
        if dict_params.__contains__('imbalance'):
            dict_params.pop('imbalance')
        dict_params['Model'] = model
        dict_params['project_name'] = project_name
        dict_params['train_project'] = train_name
        dict_params['test_project'] = test_name
        dict_params['f1-score'] = f_1
        dict_params['mcc'] = mcc
        dict_params['auc'] = auc
        dict_params['time_stamp'] = time.strftime('%m.%d/%H:%M:%S', time.localtime(time.time()))
        if not os.path.exists(gv.result_path):
            os.mkdir(gv.result_path)
        result_path = os.path.join(gv.result_path, '%s.csv' % sheet_name)
        if not os.path.exists(result_path):
            df = pd.DataFrame([dict_params], index=None)
            df.to_csv(result_path, index=False)
        else:
            df = pd.read_csv(result_path)
            df = df.append([dict_params], ignore_index=True)
            df.to_csv(result_path, index=False)
        print('finish persistence')