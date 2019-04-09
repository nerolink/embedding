from common.GlobalVariable import GlobalVariable as gv
from common.utils import batch_getter, extract_hand_craft_file_name_with_label, print_result, padding_for_vec_batch
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

logging.basicConfig(format=gv.config['logging_format'], level=gv.config['logging_level'])


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
    return dp_cnn_model


def padding_sparse(file_vec, desire_length):
    rows = []
    cols = []
    v = []
    for i in range(len(file_vec)):
        for j in range(len(file_vec[0])):
            rows.append(i)
            cols.append(j)
            v.append(file_vec[i][j])
    return sparse.FloatTensor(torch.LongTensor([rows, cols]), torch.FloatTensor(v),
                              torch.Size([desire_length, len(file_vec[0])]))


def get_file_data_p(project_name, path, set_file, dict_file_label, dict_file_hand_craft):
    """
    :param dict_file_hand_craft
    :param project_name
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
        logging.warning('load cache success in %s' % cache_name)
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


def generator_xy(batch_size, desire_length, data_x, data_y):
    """
    返回x，y的generator
    :param desire_length:
    :param batch_size:
    :param data_x:
    :param data_y:
    :return:
    """
    length = len(data_x)
    count = 0
    step = 0
    while True:
        if count >= length:
            count = 0
            step = 0
        end = count + batch_size if count + batch_size <= length else length
        print('----> batch:%d ' % step)
        yield np.array(padding_for_vec_batch(data_x[count:end], desire_length)), np.array(data_y[count:end])
        step += 1
        count = end


def generator_xhy(batch_size, desire_length, data_x, data_h, data_y):
    length = len(data_x)
    count = 0
    step = 0
    while True:
        if count >= length:
            count = 0
            step = 0
        end = count + batch_size if count + batch_size <= length else length
        print('----> batch:%d ' % step)
        yield np.array(padding_for_vec_batch(data_x[count:end], desire_length)), np.array(data_h[count:end]), np.array(
            data_y[count:end])
        step += 1
        count = end


def train_and_test_cnn(project_name, train_name, test_name):
    train_data_x, _, train_data_y, test_data_x, _, test_data_y = \
        get_train_and_test_data(project_name, train_name, test_name)
    gv.load_token_vec_length(project_name)
    cnn_model = get_cnn(gv.w2v_cnn_params)

    for epoch in range(gv.w2v_cnn_params['epochs']):
        print('epoch:%d ' % epoch)
        for step, (x, y) in enumerate(batch_getter(gv.w2v_cnn_params['batch_size'], train_data_x, train_data_y)):
            print('----> batch:%d ' % step)
            x = padding_for_vec_batch(x, gv.w2v_cnn_params['token_vec_length'])
            x = np.array(x)
            cnn_model.train_on_batch([x], y)
            del x
    # cnn_model.fit_generator(
    #     generator=generator_xy(gv.w2v_cnn_params['batch_size'], gv.w2v_cnn_params['token_vec_length'], train_data_x,
    #                            train_data_y), epochs=gv.w2v_cnn_params['epochs'],
    #     steps_per_epoch=gv.get_steps_per_epoch(len(train_data_y)))

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
    import keras.backend as k
    k.clear_session()


def train_and_test_cnn_p(project_name, train_name, test_name):
    train_data_x, train_data_hand_craft, train_data_y, test_data_x, test_data_hand_craft, test_data_y = \
        get_train_and_test_data(project_name, train_name, test_name)
    gv.load_token_vec_length(project_name)
    cnn_model_p = get_cnn_plus(gv.w2v_cnn_params)
    for epoch in range(gv.w2v_cnn_params['epochs']):
        print('epoch:%d ' % epoch)
        for step, (x, hc, y) in enumerate(
                batch_getter(gv.w2v_cnn_params['batch_size'], train_data_x, train_data_hand_craft, train_data_y)):
            print('----> batch:%d ' % step)
            x = padding_for_vec_batch(x, gv.w2v_cnn_params['token_vec_length'])
            cnn_model_p.train_on_batch([x, hc], y)
            del x
    del train_data_x
    del train_data_hand_craft
    del train_data_y
    p_y = np.array([])
    for step, (x, hc, y) in enumerate(
            batch_getter(gv.w2v_cnn_params['batch_size'], test_data_x, test_data_hand_craft, test_data_y)):
        x = padding_for_vec_batch(x, gv.w2v_cnn_params['token_vec_length'])
        _result = cnn_model_p.predict_on_batch([x, hc])
        _result = _result.squeeze()
        p_y = np.hstack((_result, p_y))
    p_y = np.array(p_y, dtype=np.float64)
    print_result(y_true=test_data_y, y_pred=p_y, dict_param=gv.w2v_cnn_params, project_name=project_name,
                 train_name=train_name, test_name=test_name, model='cnn_plus_w2v', sheet_name='cnn_w2v')
    import keras.backend as k
    k.clear_session()


def get_train_and_test_data(project_name, train_name, test_name):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    csv_train_path = gv.csv_dir + train_name + '.csv'
    csv_test_path = gv.csv_dir + test_name + '.csv'
    source_train_path = gv.projects_source_dir + train_name
    source_test_path = gv.projects_source_dir + test_name
    train_file, test_file, train_label, test_label, train_hand_craft, test_hand_craft = \
        extract_hand_craft_file_name_with_label(csv_train_path,
                                                csv_test_path)
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
    train_and_test_cnn('camel', 'camel-1.2', 'camel-1.4')
