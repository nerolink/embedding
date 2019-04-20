from common.GlobalVariable import instance as global_var
from common.utils import batch_getter, extract_hand_craft_file_name_with_label, print_result, padding_for_vec_batch, \
    z_score
import Huffman as hf
import HierarchicalSoftmax as hs
from keras.models import Model
import numpy as np
from torch import sparse
import torch
from keras import Sequential
from keras.layers import *
import pickle
import os
import javalang as jl
import logging

logging.basicConfig(format=global_var.config['logging_format'], level=global_var.config['logging_level'])


def get_cnn(dict_params):
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'],
                         input_shape=(dict_params['token_vec_length'], dict_params['vec_size']), activation='relu'))
    # cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    # cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    # cnn_model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    # cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    # cnn_model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=dict_params['hidden_units'], activation='relu'))
    cnn_model.add(Dense(units=1, activation='sigmoid'))
    cnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=dict_params['metrics'])
    return cnn_model


def get_cnn_plus(dict_params):
    """
    :param dict_params: 模型参数字典
    :return:    编译好的模型
    """

    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'],
                         input_shape=(dict_params['token_vec_length'], dict_params['vec_size']), activation='relu'))
    # cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    # cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    # cnn_model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    # cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    # cnn_model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=dict_params['hidden_units'], activation='relu'))

    hand_craft_model = Sequential()
    i = InputLayer(input_shape=(dict_params['hand_craft_input_dim'],))
    hand_craft_model.add(i)
    concatenate = Concatenate()([cnn_model.output, hand_craft_model.output])
    out = Dense(1, activation='sigmoid')(concatenate)
    dp_cnn_model = Model(inputs=[cnn_model.input, hand_craft_model.input], outputs=out)
    dp_cnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=dict_params['metrics'])
    # dp_cnn_model.compile(loss='binary_crossentropy', optimizer='adams', metrics=dict_params['metrics'])
    return dp_cnn_model


def get_file_data_p(project_name, path, set_file, dict_file_label, dict_file_hand_craft, gv=global_var):
    """
    获取一个项目的标签，手工标注特征，完整类名，word2vec表示
    :param gv:
    :param dict_file_hand_craft
    :param project_name 项目源文件
    :param path:
    :param set_file:
    :param dict_file_label:
    :return:[[label,[hand_craft_data],full_class_name,[...],[...],]]
    """
    gv.load_word2vec(project_name)
    gv.load_token_vec_length(project_name)
    method_name = 'get_file_data_p'
    cache_name = '%s_%s_%d' % (path, method_name, gv.w2v_cnn_params['vec_size'])
    result = gv.load_cache(cache_name)
    if result is not None:
        logging.info('load cache success in %s' % cache_name)
        return result
    result = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            full_class_name = hf.get_full_class_name(root, file_name)
            if full_class_name in set_file:
                file_obj = open(os.path.join(root, file_name), 'r', encoding='utf-8')
                input_data = []  # 一个文件的embedding后的token_vec [[],[],...,label]
                ('processing file %s in get_file_data_p' % os.path.join(root, file_name))
                try:
                    ast_tree = jl.parse.parse(file_obj.read())
                    for path, node in ast_tree:
                        node_name = hf.get_node_name(node)
                        if node_name is not None:
                            input_data.append(gv.word_to_vec[node_name].cpu().numpy().tolist())
                    # padding(input_data, gv.params['token_vec_length'])
                    input_data.insert(0, dict_file_hand_craft[full_class_name])
                    input_data.insert(0, full_class_name)
                    input_data.insert(0, dict_file_label[full_class_name])
                    result.append(input_data)
                except jl.parser.JavaSyntaxError:
                    logging.error('parse file %s error' % os.path.join(root, file_name))
                except AttributeError:
                    logging.error('parse file %s attribute error in get_file_data_p ' % os.path.join(root, file_name))
                except UnicodeDecodeError:
                    logging.error('parse file %s unicode decode error' % os.path.join(root, file_name))
                finally:
                    file_obj.close()
    gv.dump_cache(cache_name, result)
    return result


def train_and_test_cnn(project_name, train_name, test_name, gv=global_var):
    train_data_x, _, train_data_y, test_data_x, _, test_data_y = \
        get_train_and_test_data(project_name, train_name, test_name)
    gv.load_token_vec_length(project_name)

    model_name = '%s~~%d~~%s' % (train_name, gv.w2v_cnn_params['vec_size'], 'cnn_w2v')
    cnn_model = gv.load_model(model_name)
    if cnn_model is None:
        cnn_model = get_cnn(gv.w2v_cnn_params)
        for epoch in range(gv.w2v_cnn_params['epochs']):
            print('epoch:%d ' % epoch)
            for step, (x, y) in enumerate(batch_getter(gv.w2v_cnn_params['batch_size'], train_data_x, train_data_y)):
                print('----> batch:%d ' % step)
                x = padding_for_vec_batch(x, gv.w2v_cnn_params['token_vec_length'])
                x = np.array(x)
                cnn_model.train_on_batch([x], y)
                del x
        gv.dump_model(cnn_model, model_name)

    del train_data_x
    del train_data_y
    p_y = np.array([])
    for step, (x, y) in enumerate(
            batch_getter(gv.w2v_cnn_params['batch_size'], test_data_x, test_data_y)):
        x = padding_for_vec_batch(x, gv.w2v_cnn_params['token_vec_length'])
        x = np.array(x)
        _result = cnn_model.predict_on_batch(x)
        _result = _result.squeeze()
        p_y = np.hstack((_result, p_y))
    p_y = np.array(p_y, dtype=np.float64)
    print_result(y_true=test_data_y, y_pred=p_y, dict_param=gv.w2v_cnn_params, project_name=project_name,
                 train_name=train_name, test_name=test_name, model='cnn_w2v', sheet_name='cnn_w2v')
    gv.w2v_cnn_params['train_project'] = train_name
    gv.w2v_cnn_params['test_project'] = test_name
    import keras.backend as k
    k.clear_session()


def train_and_test_cnn_p(project_name, train_name, test_name, gv=global_var):
    train_data_x, train_data_hand_craft, train_data_y, test_data_x, test_data_hand_craft, test_data_y = \
        get_train_and_test_data(project_name, train_name, test_name)
    gv.load_token_vec_length(project_name)
    model_name = '%s~~%d~~%s' % (train_name, gv.w2v_cnn_params['vec_size'], 'cnn_w2v')
    cnn_model = gv.load_model(model_name)
    if cnn_model is None:
        cnn_model = get_cnn(gv.w2v_cnn_params)
        for epoch in range(gv.w2v_cnn_params['epochs']):
            print('epoch:%d ' % epoch)
            for step, (x, y) in enumerate(batch_getter(gv.w2v_cnn_params['batch_size'], train_data_x, train_data_y)):
                print('----> batch:%d ' % step)
                x = padding_for_vec_batch(x, gv.w2v_cnn_params['token_vec_length'])
                x = np.array(x)
                cnn_model.train_on_batch([x], y)
                inter_model = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer(index=3).output)
                print(np.array(inter_model.predict(x)).shape)
                del x
        gv.dump_model(cnn_model, model_name)

    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(solver='lbfgs', max_iter=1000)
    inter_model = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer(index=3).output)
    cls_train_data = None
    for step, (x, hc) in enumerate(
            batch_getter(gv.w2v_cnn_params['batch_size'], train_data_x, train_data_hand_craft)):
        print('----> batch:%d ' % step)
        x = padding_for_vec_batch(x, gv.w2v_cnn_params['token_vec_length'])
        x = np.array(x)
        x = inter_model.predict(x)
        # x = z_score(x)
        hc = np.array(hc)
        cls_train_data = np.hstack((x, hc)) if cls_train_data is None else np.vstack(
            (cls_train_data, np.hstack((x, hc))))
        del x

    cls.fit(cls_train_data, np.array(train_data_y))
    del train_data_x
    del train_data_hand_craft
    del train_data_y
    p_y = np.array([])
    for step, (x, hc, y) in enumerate(
            batch_getter(gv.w2v_cnn_params['batch_size'], test_data_x, test_data_hand_craft, test_data_y)):
        x = padding_for_vec_batch(x, gv.w2v_cnn_params['token_vec_length'])
        x = np.array(x)
        x = inter_model.predict(x)
        # x = z_score(x)
        hc = np.array(hc)
        _result = cls.predict(np.hstack((x, hc)))
        _result = _result.squeeze()
        p_y = np.hstack((_result, p_y))
    p_y = np.array(p_y, dtype=np.float64)
    print_result(y_true=test_data_y, y_pred=p_y, dict_param=gv.w2v_cnn_params, project_name=project_name,
                 train_name=train_name, test_name=test_name, model='cnn_plus_w2v', sheet_name='cnn_w2v')
    gv.w2v_cnn_params['train_project'] = train_name
    gv.w2v_cnn_params['test_project'] = test_name
    import keras.backend as k
    k.clear_session()


def get_train_and_test_data(project_name, train_name, test_name, gv=global_var):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    csv_train_path = gv.csv_dir + train_name + '.csv'
    csv_test_path = gv.csv_dir + test_name + '.csv'
    source_train_path = gv.projects_source_dir + train_name
    source_test_path = gv.projects_source_dir + test_name
    train_file, test_file, train_label, test_label, train_hand_craft, test_hand_craft = \
        extract_hand_craft_file_name_with_label(csv_train_path, csv_test_path)
    hs.train(project_name)
    train_data = get_file_data_p(project_name, source_train_path, set(train_file),
                                 dict(zip(train_file, train_label)), dict(zip(train_file, train_hand_craft)))
    # [[],[],...[hand_craft_data],full_class_name] ,
    _features = np.array([x[1:] for x in train_data]).reshape(-1, 1)
    _labels = np.array([x[0] for x in train_data])
    gv.w2v_cnn_params['raw_train_size'] = len(_labels)
    _features, _labels = ros.fit_resample(_features, _labels)
    _features = _features.squeeze().tolist()
    _labels = _labels.squeeze().tolist()
    gv.w2v_cnn_params['train_size'] = len(_labels)
    train_data_x = [x[2:] for x in _features]
    train_data_y = _labels
    train_data_hand_craft = [x[1] for x in _features]
    test_data = get_file_data_p(project_name, source_test_path, set(test_file), dict(zip(test_file, test_label)),
                                dict(zip(test_file, test_hand_craft)))
    _features = [x[1:] for x in test_data]
    _labels = [x[0] for x in test_data]
    gv.w2v_cnn_params['test_size'] = len(_labels)
    test_data_x = [x[2:] for x in _features]
    test_data_y = _labels
    test_data_hand_craft = [x[1] for x in _features]
    return train_data_x, train_data_hand_craft, train_data_y, test_data_x, test_data_hand_craft, test_data_y


if __name__ == '__main__':
    import common.GlobalVariable as cgv
    for pn, sources in cgv.projects.items():
        for i in range(len(sources) - 1):
            print("train name %s,test name %s" % (sources[i], sources[i + 1]))
            tax, _, _, _, _, _ = get_train_and_test_data(pn, sources[i], sources[i + 1])
            print(len(tax))
