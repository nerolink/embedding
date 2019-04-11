import os

from keras import Sequential
from keras.layers import *
from keras.models import Model
from common.GlobalVariable import GlobalVariable as gv
from common.GlobalVariable import projects as pj
from common.utils import get_md_data, print_result, batch_getter, padding_for_token_batch


def get_cnn_model(dict_params):
    """
    :param dict_params: 模型参数字典
    :return:    编译好的模型
    """

    cnn_model = Sequential()
    cnn_model.add(Embedding(input_dim=dict_params['input_dim'], output_dim=dict_params['output_dim'],
                            input_length=dict_params['input_length']))
    cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=dict_params['hidden_units'], activation='relu'))
    cnn_model.add(Dense(units=1, activation='sigmoid'))
    cnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=dict_params['metrics'])
    return cnn_model


def get_cnn_model_p(dict_params):
    """
    :param dict_params: 模型参数字典
    :return:    编译好的模型
    """

    cnn_model = Sequential()
    cnn_model.add(Embedding(input_dim=dict_params['input_dim'], output_dim=dict_params['output_dim'],
                            input_length=dict_params['input_length']))
    cnn_model.add(Conv1D(filters=dict_params['filters'], kernel_size=dict_params['kernel_size'], activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=dict_params['pool_size'], padding='valid'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=dict_params['hidden_units'], activation='relu'))
    hand_craft_model = Sequential()
    i = InputLayer(input_shape=(dict_params['hand_craft_input_dim'],))
    hand_craft_model.add(i)
    _concatenate = Concatenate()([cnn_model.output, hand_craft_model.output])
    out = Dense(1, activation='sigmoid')(_concatenate)
    cnn_model_p = Model(inputs=[cnn_model.input, hand_craft_model.input], outputs=out)
    cnn_model_p.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=dict_params['metrics'])
    return cnn_model_p


def train_and_test_cnn(project_name, train_name, test_name, dict_params):
    """
    :param project_name:
    :param train_name:
    :param test_name:
    :param dict_params:
    :return:
    """
    train_source_path = os.path.join(gv.projects_source_dir, train_name)
    test_source_path = os.path.join(gv.projects_source_dir, test_name)
    train_csv_path = os.path.join(gv.csv_dir, train_name)
    test_csv_path = os.path.join(gv.csv_dir, test_name)
    [train_data, test_data] = get_md_data([[train_source_path, train_csv_path],
                                           [test_source_path, test_csv_path]], dict_params['imbalance'])

    _train_x = train_data.get_ast_vectors()
    _train_y = train_data.get_labels()
    _test_x = test_data.get_ast_vectors()
    _test_y = test_data.get_labels()
    dict_params['input_length'] = max(train_data.get_vec_len(), test_data.get_vec_len())
    model_name = '%s~~%s' % (train_name, 'cnn_plain')
    _model = gv.load_model(model_name)
    if _model is None:
        _model = get_cnn_model(dict_params)
        for epoch in range(dict_params['epochs']):
            print('epoch:%d ' % epoch)
            for step, (x, y) in enumerate(batch_getter(dict_params['batch_size'], _train_x, _train_y)):
                x = padding_for_token_batch(x, dict_params['input_length'])
                _model.train_on_batch([x], y)
        gv.dump_model(_model, model_name)

    _y_predict = []
    for step, (x, y) in enumerate(batch_getter(dict_params['batch_size'], _test_x, _test_y)):
        x = padding_for_token_batch(x, dict_params['input_length'])
        _y_predict += _model.predict_on_batch([x]).squeeze().tolist()

    _model.fit(x=[_train_x], y=_train_y, epochs=dict_params['epochs'], batch_size=dict_params['batch_size'])
    _y_predict = _model.predict([_test_x], batch_size=dict_params['batch_size'])
    print_result(y_true=_test_y, y_pred=_y_predict, dict_param=dict_params, project_name=project_name,
                 train_name=train_name, test_name=test_name, model='cnn_plain', sheet_name='cnn_plain')
    import keras.backend as k
    k.clear_session()


def train_and_test_cnn_p(project_name, train_name, test_name, dict_params):
    """
    :param project_name:
    :param train_name:
    :param test_name:
    :param dict_params:
    :return:
    """

    train_source_path = os.path.join(gv.projects_source_dir, train_name)
    test_source_path = os.path.join(gv.projects_source_dir, test_name)
    train_csv_path = os.path.join(gv.csv_dir, train_name)
    test_csv_path = os.path.join(gv.csv_dir, test_name)
    [train_data, test_data] = get_md_data([[train_source_path, train_csv_path],
                                           [test_source_path, test_csv_path]], dict_params['imbalance'])

    _train_x = train_data.get_ast_vectors()
    _train_y = train_data.get_labels()
    _train_hc = train_data.get_hand_craft_vectors()
    _test_hc = test_data.get_hand_craft_vectors()
    _test_x = test_data.get_ast_vectors()
    _test_y = test_data.get_labels()
    dict_params['input_length'] = max(train_data.get_vec_len(), test_data.get_vec_len())
    model_name = '%s~~%s' % (train_name, 'cnn_plus_plain')

    _model = gv.load_model(model_name)
    if _model is None:
        _model = get_cnn_model_p(dict_params)
        for epoch in range(dict_params['epochs']):
            print('epoch:%d ' % epoch)
            for step, (x, hc, y) in enumerate(batch_getter(dict_params['batch_size'], _train_x, _train_hc, _train_y)):
                print('batch------- %s' % step)
                x = padding_for_token_batch(x, dict_params['input_length'])
                _model.train_on_batch(x=[x, hc], y=y)
                del x
        gv.dump_model(_model, model_name)

    _y_predict = np.array([])
    for step, (x, hc) in enumerate(batch_getter(dict_params['batch_size'], _test_x, _test_hc)):
        x = padding_for_token_batch(x, dict_params['input_length'])
        tmp = _model.predict_on_batch([x, hc]).squeeze()
        _y_predict = np.hstack((_y_predict, tmp))
    print_result(y_true=_test_y, y_pred=_y_predict, dict_param=dict_params, project_name=project_name,
                 train_name=train_name, test_name=test_name, model='cnn_plain_plus', sheet_name='cnn_plain')
    import keras.backend as k
    k.clear_session()


def train_and_test_cnn_p_copy(project_name, train_name, test_name, dict_params):
    """
    :param project_name:
    :param train_name:
    :param test_name:
    :param dict_params:
    :return:
    """

    train_source_path = os.path.join(gv.projects_source_dir, train_name)
    test_source_path = os.path.join(gv.projects_source_dir, test_name)
    train_csv_path = os.path.join(gv.csv_dir, train_name)
    test_csv_path = os.path.join(gv.csv_dir, test_name)
    [train_data, test_data] = get_md_data([[train_source_path, train_csv_path],
                                           [test_source_path, test_csv_path]], dict_params['imbalance'])
    dict_params['input_length'] = train_data.get_vec_len()
    _model = get_cnn_model(dict_params)
    _model.fit_generator(train_data.generator_get_data_xy(dict_params['batch_size']),
                         epochs=dict_params['epochs'],
                         steps_per_epoch=dict_params['batch_size'])
    _y_predict = _model.predict_generator(test_data.generator_get_data_x(dict_params['batch_size']),
                                          (test_data.get_data_size() + dict_params['batch_size'] - 1) / dict_params[
                                              'batch_size'])
    # _y_predict = _model.predict_generator(test_data.generator_get_data_x(dict_params['batch_size']))
    _test_y = test_data.get_labels()
    print_result(y_true=_test_y, y_pred=_y_predict, dict_param=dict_params, project_name=project_name,
                 train_name=train_name, test_name=test_name, model='cnn_plain_plus', sheet_name='cnn_plain')
    import keras.backend as k
    k.clear_session()


if __name__ == '__main__':
    train_and_test_cnn('jedit', pj['jedit'][2], pj['jedit'][3], gv.plain_cnn_params)
