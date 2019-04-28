import os

from keras import Sequential
from keras.layers import *
from keras.models import Model
from sklearn.linear_model import LogisticRegression

from common.GlobalVariable import instance as global_var
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


def train_and_test_cnn(project_name, train_name, test_name, dict_params, gv=global_var):
    """
    :param gv:
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
    print_result(y_true=_test_y, y_pred=_y_predict, dict_params=dict_params, project_name=project_name,
                 train_name=train_name, test_name=test_name, model='cnn_plain', sheet_name='cnn_plain')
    import keras.backend as k
    k.clear_session()


def train_and_test_cnn_p(project_name, train_name, test_name, dict_params, gv=global_var):
    """
    :param gv:
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
    cls = LogisticRegression(solver='lbfgs', max_iter=1000)
    if _model is None:
        _model = get_cnn_model(dict_params)
        for epoch in range(dict_params['epochs']):
            print('epoch:%d ' % epoch)
            for step, (x, hc, y) in enumerate(batch_getter(dict_params['batch_size'], _train_x, _train_hc, _train_y)):
                print('batch------- %s' % step)
                x = padding_for_token_batch(x, dict_params['input_length'])
                _model.train_on_batch(x=[x], y=y)
                del x
        gv.dump_model(_model, model_name)
    cls.fit(X=_train_hc, y=_train_y)
    final_cls = LogisticRegression(solver='lbfgs', max_iter=1000)
    final_data = np.array([])
    for step, (x, y) in enumerate(batch_getter(gv.w2v_cnn_params['batch_size'], _train_x, _train_y)):
        print('----> batch:%d ' % step)
        x = padding_for_token_batch(x, dict_params['input_length'])
        x = np.array(x)
        p_cnn = _model.predict(x).reshape(-1)
        final_data = np.hstack((p_cnn, final_data))
    final_data = final_data.reshape((len(final_data), 1))
    final_data = np.hstack((final_data, cls.predict(X=_train_hc).reshape((len(final_data), 1))))
    final_cls.fit(X=final_data, y=_train_y)

    p_y = np.array([])
    for step, (x, hc) in enumerate(batch_getter(dict_params['batch_size'], _test_x, _test_hc)):
        x = padding_for_token_batch(x, dict_params['input_length'])
        x = np.array(x)
        p_cnn = _model.predict(x).reshape((len(x), 1))
        p_hc = cls.predict(np.array(hc)).reshape((len(x), 1))
        _result = final_cls.predict(X=np.hstack((p_cnn, p_hc))).squeeze()
        p_y = np.hstack((p_y, _result))

    print_result(y_true=_test_y, y_pred=p_y, dict_params=dict_params, project_name=project_name,
                 train_name=train_name, test_name=test_name, model='cnn_plain_plus', sheet_name='cnn_plain')
    import keras.backend as k
    k.clear_session()


# def train_and_test_cnn_p(project_name, train_name, test_name, dict_params, gv=global_var):
#     """
#     :param gv:
#     :param project_name:
#     :param train_name:
#     :param test_name:
#     :param dict_params:
#     :return:
#     """
#     train_source_path = os.path.join(gv.projects_source_dir, train_name)
#     test_source_path = os.path.join(gv.projects_source_dir, test_name)
#     train_csv_path = os.path.join(gv.csv_dir, train_name)
#     test_csv_path = os.path.join(gv.csv_dir, test_name)
#     [train_data, test_data] = get_md_data([[train_source_path, train_csv_path],
#                                            [test_source_path, test_csv_path]], dict_params['imbalance'])
#
#     _train_x = train_data.get_ast_vectors()
#     _train_y = train_data.get_labels()
#     _train_hc = train_data.get_hand_craft_vectors()
#     _test_hc = test_data.get_hand_craft_vectors()
#     _test_x = test_data.get_ast_vectors()
#     _test_y = test_data.get_labels()
#     dict_params['input_length'] = max(train_data.get_vec_len(), test_data.get_vec_len())
#     model_name = '%s~~%s' % (train_name, 'cnn_plus_plain')
#     _model = gv.load_model(model_name)
#     if _model is None:
#         _model = get_cnn_model(dict_params)
#         for epoch in range(dict_params['epochs']):
#             print('epoch:%d ' % epoch)
#             for step, (x, y) in enumerate(batch_getter(dict_params['batch_size'], _train_x, _train_y)):
#                 print('batch------- %s' % step)
#                 x = padding_for_token_batch(x, dict_params['input_length'])
#                 _model.train_on_batch(x=[x], y=y)
#                 del x
#         gv.dump_model(_model, model_name)
#     inter_model = Model(inputs=_model.input, outputs=_model.get_layer(index=4).output)
#     cls = LogisticRegression(solver='lbfgs', max_iter=1000)
#     cls_train_data = None
#     for step, (x, hc) in enumerate(
#             batch_getter(gv.w2v_cnn_params['batch_size'], _train_x, _train_hc)):
#         print('----> batch:%d ' % step)
#         x = padding_for_token_batch(x, dict_params['input_length'])
#         x = np.array(x)
#         x = inter_model.predict(x)
#         # x = z_score(x)
#         hc = np.array(hc)
#         cls_train_data = np.hstack((x, hc)) if cls_train_data is None else np.vstack(
#             (cls_train_data, np.hstack((x, hc))))
#         del x
#     cls.fit(cls_train_data, np.array(_train_y))
#     p_y = np.array([])
#     for step, (x, hc) in enumerate(batch_getter(dict_params['batch_size'], _test_x, _test_hc)):
#         x = padding_for_token_batch(x, dict_params['input_length'])
#         x = np.array(x)
#         x = inter_model.predict(x)
#         hc = np.array(hc)
#         _result = cls.predict(np.hstack((x, hc)))
#         _result = _result.squeeze()
#         p_y = np.hstack((_result, p_y))
#
#     print_result(y_true=_test_y, y_pred=p_y, dict_params=dict_params, project_name=project_name,
#                  train_name=train_name, test_name=test_name, model='cnn_plain_plus', sheet_name='cnn_plain')
#     import keras.backend as k
#     k.clear_session()

def train_and_test_cnn_p_copy(project_name, train_name, test_name, dict_params, gv=global_var):
    """
    :param gv:
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
    print_result(y_true=_test_y, y_pred=_y_predict, dict_params=dict_params, project_name=project_name,
                 train_name=train_name, test_name=test_name, model='cnn_plain_plus', sheet_name='cnn_plain')
    import keras.backend as k
    k.clear_session()


if __name__ == '__main__':
    train_and_test_cnn('jedit', pj['jedit'][2], pj['jedit'][3], global_var.plain_cnn_params)
