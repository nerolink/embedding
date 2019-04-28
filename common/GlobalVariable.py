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
import keras.models as km
import threading

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
global_lock = {'camel': threading.Lock(), 'forrest': threading.Lock(), 'jedit': threading.Lock(),
               'log4j': threading.Lock(), 'lucene': threading.Lock(), 'poi': threading.Lock(),
               'synapse': threading.Lock(), 'velocity': threading.Lock(), 'xalan': threading.Lock(),
               'xerces': threading.Lock(), }
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
    'camel': ['camel-1.2', 'camel-1.4', 'camel-1.6'],
    'forrest': ['forrest-0.6', 'forrest-0.7', 'forrest-0.8'],
    'jedit': ['jedit-3.2.1', 'jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3'],
    # 'ivy': ['ivy-1.1', 'ivy-1.4', 'ivy-2.0'],
    'log4j': ['log4j-1.0', 'log4j-1.1', 'log4j-1.2'],
    'lucene': ['lucene-2.0', 'lucene-2.2', 'lucene-2.4'],
    'poi': ['poi-1.5', 'poi-2.0', 'poi-2.5.1', 'poi-3.0'],
    'synapse': ['synapse-1.0', 'synapse-1.1', 'synapse-1.2'],
    'velocity': ['velocity-1.4', 'velocity-1.5', 'velocity-1.6.1'],
    'xalan': ['xalan-2.4', 'xalan-2.5', 'xalan-2.6', 'xalan-2.7'],
    'xerces': ['xerces-1.2', 'xerces-1.3', 'xerces-1.4.4']
}


# projects = {
#     'forrest': ['forrest-0.6', 'forrest-0.7'],
#     'jedit': ['jedit-4.2', 'jedit-4.3'],
#     'log4j': ['log4j-1.1', 'log4j-1.2'],
#     'poi': ['poi-1.5', 'poi-2.0'],
#     'velocity': ['velocity-1.4', 'velocity-1.5'],
#     'xalan': ['xalan-2.4', 'xalan-2.5'],
#     'xerces': ['xerces-1.2', 'xerces-1.3']
# }
candidate = {
    # 'vec_size': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100],
    'vec_size': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    'number_of_filter': [10, 20, 50, 100, 150, 200],
    'filter_length': [2, 3, 5, 10, 20, 50, 100],
    'hidden_unit': [10, 20, 30, 50, 100, 150, 200, 250]
}


class GlobalVariable:

    def __init__(self):

        self.word_to_vec = {}
        self.word_to_node = {}
        self.d_type = torch.float64
        # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        # 用于四舍五入
        self.round_threshold = 0.5

        # #for test
        # w2v_cnn_params = {'Model': 'cnn', 'vec_size': 50, 'learning_rate': 0.01, 'round_threshold': 0.5,
        #                   'token_vec_length': 0, 'batch_size': 32, 'filters': 50, 'kernel_size': 20, 'mcc': None,
        #                   'hand_craft_input_dim': 20, 'pool_size': 2, 'hidden_units': 150, 'epochs': 1, 'metrics': ['acc'],
        #                   'project_name': None, 'train_project': None, 'test_project': None, 'time_stamp': None,
        #                   'auc': None, 'f1-score': None, 'use_cuda': True}

        self.w2v_cnn_params = {'Model': 'cnn', 'vec_size': 16, 'learning_rate': 0.01, 'round_threshold': 0.5,
                               'token_vec_length': 0, 'batch_size': 32, 'filters': 10, 'kernel_size': 5, 'mcc': None,
                               'hand_craft_input_dim': 20, 'pool_size': 2, 'hidden_units': 100, 'epochs': 15,
                               'metrics': ['acc'], 'project_name': None, 'train_project': None, 'test_project': None,
                               'time_stamp': None, 'auc': None, 'f1-score': None, 'use_cuda': True}

        self.plain_cnn_params = {'input_dim': 3709, 'output_dim': 30, 'input_length': 2405, 'filters': 10,
                                 'kernel_size': 5, 'pool_size': 2, 'hidden_units': 100, 'hand_craft_input_dim': 20,
                                 'metrics': ['acc'], 'batch_size': 32, 'epochs': 15, 'imbalance': RandomOverSampler(),
                                 'regenerate': False}

        self.dbn_params = {'output_size': 100, 'hidden_layer_num': 3, 'epochs': 1, 'batch_size': 10,
                           'imbalance': RandomOverSampler(), 'learning_rate': 0.1, 'regenerate': False}

        self.config = {'logging_level': logging.INFO,
                       'logging_format': '%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                       'remake': False}

        self.metrics = ['acc']
        # self.projects_source_dir = "J:\\sdp\\projects\\"
        # self.csv_dir = "J:\\sdp\\csvs\\"

        self.projects_source_dir = "M:\\sdp\\projects\\"
        self.csv_dir = "M:\\sdp\\csvs\\"
        self.hf_root = None
        self.training_data = []
        self.debug_map = {}
        self.count = 0
        # 设置为True表示不使用缓存
        self.isDebug = False
        self.requires_grad = False
        self.sigmoid_threshold = 6
        self.data_path = '../data/'
        self.data_cache = self.data_path + 'cache'
        self.hf_tree_path = self.data_path + 'hf_tree'
        self.token_len_path = self.data_path + 'token_vec_len'
        self.word_to_vec_path = self.data_path + 'word_to_vec'
        self.result_path = self.data_path + 'result'
        self.model_path = self.data_path + 'model'
        self.picture_path = self.data_path + 'picture'
        self.w2v_result_path = self.result_path + '/cnn_w2v.csv'
        self.plain_cnn_result_path = self.result_path + '/cnn_plain.csv'
        self.dbn_result_path = self.result_path + '/dbn.csv'
        self.lr_result_path = self.result_path + '/LogisticRegression.csv'
        self.current_project = None
        self.last_result_w2v = 'D:\\OneDrive\\document\\paper\\result\\cnn_w2v.csv'
        self.last_result_cnn = 'D:\\OneDrive\\document\\paper\\result\\cnn_plain.csv'
        self.last_result_dbn = 'D:\\OneDrive\\document\\paper\\result\\dbn.csv'

    def get_steps_per_epoch(self, data_size):
        return (data_size + self.w2v_cnn_params['batch_size'] - 1) / self.w2v_cnn_params[
            'batch_size']

    def load_hf_tree(self, project_name):
        if project_name is None or self.config['remake']:
            return None
        cache_dir = os.path.join(self.hf_tree_path, '%s.ht' % project_name)
        if not os.path.exists(cache_dir):
            GlobalVariable.hf_root = None
            return False
        with open(cache_dir, 'rb') as file_obj:
            self.hf_root = pickle.load(file_obj)
        self.current_project = project_name
        return True

    def dump_hf_tree(self, project_name, hf_tree):
        if project_name is None:
            return None
        if not os.path.exists(self.hf_tree_path):
            os.mkdir(self.hf_tree_path)
        cache_dir = os.path.join(self.hf_tree_path, '%s.ht' % project_name)
        with open(cache_dir, 'wb') as file_obj:
            pickle.dump(hf_tree, file_obj)

    def load_token_vec_length(self, project_name):

        if project_name is None or self.config['remake'] or self.isDebug:
            return None
        tvl_path = os.path.join(self.token_len_path, '%s.tvl' % project_name)
        if not os.path.exists(tvl_path):
            self.w2v_cnn_params['token_vec_length'] = 0
            return False
        with open(tvl_path, 'rb') as file_obj:
            self.w2v_cnn_params['token_vec_length'] = pickle.load(file_obj)
        self.current_project = project_name
        return True

    def dump_token_vec_length(self, project_name):
        if project_name is None:
            return None
        if not os.path.exists(self.token_len_path):
            os.mkdir(self.token_len_path)
        tvl_path = os.path.join(self.token_len_path, '%s.tvl' % project_name)
        with open(tvl_path, 'wb') as file_obj:
            pickle.dump(self.w2v_cnn_params['token_vec_length'], file_obj)

    def load_word2vec(self, project_name):

        if project_name is None or self.config['remake'] or self.isDebug:
            return None
        w2v_path = os.path.join(self.word_to_vec_path,
                                '%s_%d.w2v' % (project_name, self.w2v_cnn_params['vec_size']))
        if not os.path.exists(w2v_path):
            self.word_to_vec = {}
            return False
        with open(w2v_path, 'rb') as file_obj:
            self.word_to_vec = pickle.load(file_obj)
        self.current_project = project_name
        return True

    def dump_word2vec(self, project_name):
        """
        只和vec_size有关
        :param project_name:
        :return:
        """

        if project_name is None:
            return
        if not os.path.exists(self.word_to_vec_path):
            os.mkdir(self.word_to_vec_path)
        w2v_path = os.path.join(self.word_to_vec_path,
                                '%s_%d.w2v' % (project_name, self.w2v_cnn_params['vec_size']))
        with open(w2v_path, 'wb') as file_obj:
            pickle.dump(self.word_to_vec, file_obj)

    def dump_word2vec_txt(self, project_name):

        if project_name is None:
            return
        if not os.path.exists(self.word_to_vec_path):
            os.mkdir(self.word_to_vec_path)
        w2v_path = os.path.join(self.word_to_vec_path,
                                '%s_%d.txt' % (project_name, self.w2v_cnn_params['vec_size']))
        with open(w2v_path, 'w') as file_obj:
            for k, v in self.word_to_vec.items():
                file_obj.write(str(k) + ":" + str(list(v.data.cpu().numpy())) + "\n")
        return

    def dump_cache(self, file_name, obj):
        if file_name is None:
            return
        m2 = hashlib.md5()
        m2.update(file_name.encode('utf-8'))
        file_name = m2.hexdigest()
        cache_path = os.path.join(self.data_cache, file_name)
        if not os.path.exists(self.data_cache):
            os.mkdir(self.data_cache)
        with open(cache_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    def load_cache(self, file_name):
        if file_name is None or self.config['remake'] or self.isDebug:
            return None
        m2 = hashlib.md5()
        m2.update(file_name.encode('utf-8'))
        file_name = m2.hexdigest()
        cache_path = os.path.join(self.data_cache, file_name)
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, 'rb') as file_obj:
            result = None
            try:
                result = pickle.load(file_obj)
            except EOFError:
                result = None
            finally:
                return result

    def dump_model(self, model, model_name):
        model_path = os.path.join(self.model_path, model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        model.save(model_path)

    def load_model(self, model_name):
        model_path = os.path.join(self.model_path, model_name)
        if not os.path.exists(model_path) or self.isDebug:
            return None
        return km.load_model(model_path)

    def dump_dbn(self, dbn_model, model_name):
        model_path = os.path.join(self.model_path, model_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        with open(model_path, 'wb') as file_obj:
            pickle.dump(dbn_model, file_obj)

    def load_dbn(self, model_name):
        model_path = os.path.join(self.model_path, model_name)
        if not os.path.exists(model_path) or self.isDebug:
            return None
        with open(model_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    def persistence(self, dict_params, project_name, train_name, test_name, f_1, mcc, auc, model, y_true, y_pred,
                    sheet_name=None):
        import pandas as pd
        print('begin persistence')
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
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        result_path = os.path.join(self.result_path, '%s.csv' % sheet_name)
        if not os.path.exists(result_path):
            df = pd.DataFrame([dict_params], index=None)
            df.to_csv(result_path, index=False)
        else:
            df = pd.read_csv(result_path)
            df = df.append([dict_params], ignore_index=True)
            df.to_csv(result_path, index=False)
        print('finish persistence')


instance = GlobalVariable()
