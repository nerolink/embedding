import logging
import tensorflow as tf
import numpy as np
from common.GlobalVariable import instance as global_var
from common.utils import get_all_need_class_name, ProjectData, get_md_data, print_result, padding_for_token_batch
import warnings

warnings.simplefilter("ignore", Warning)
import os

logging.basicConfig(format=global_var.config['logging_format'], level=global_var.config['logging_level'])


class RBM(object):
    def __init__(self, input_size, output_size, name, params):

        self.w = np.random.normal(size=[input_size, output_size])
        self.vb = np.zeros([input_size])
        self.hb = np.zeros([output_size])
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.params = params

        # with tf.name_scope("rbm_" + str(name)):
        #     self.weights = tf.Variable(tf.truncated_normal(shape=[input_size, output_size], stddev=1, name="weight"))
        #     self.v_bias = tf.Variable(tf.zeros(shape=[input_size], name="v_bias"))
        #     self.h_bias = tf.Variable(tf.zeros(shape=[output_size], name="h_bias"))

    @staticmethod
    def split_batch(data_size, batch_size):
        start_indexes = []
        end_indexes = []

        if data_size < batch_size:
            start_indexes.append(0)
            end_indexes.append(data_size)
        else:
            end_index = 0
            for i, j in zip(range(0, data_size, batch_size), range(batch_size, data_size, batch_size)):
                start_indexes.append(i)
                end_indexes.append(j)
                end_index = j
            if end_index < data_size:
                start_indexes.append(end_index)
                end_indexes.append(data_size)
        return start_indexes, end_indexes

    @staticmethod
    def sample_from_probability(probability):
        """
        输入一个可见层或隐藏层概率为1向量，输出单元为0或1的向量，
        :param probability:    [0.1,0.2,0.4,0.3......]
        :return:                [0,0,1,1,.....]
        """
        return tf.nn.relu(tf.sign(probability - tf.random_uniform(shape=tf.shape(probability), minval=0, maxval=1)))

    @staticmethod
    def probability_h_to_v(hidden, weights, v_bias):
        """

        :param hidden:      隐含层的值，[1,0,0,0.....]  output_size
        :param weights
        :param v_bias
        :return:            可见层的概率  [0.1,0.2,...]  input_size
        """
        return tf.sigmoid(tf.matmul(hidden, tf.transpose(weights)) + v_bias)

    @staticmethod
    def probability_v_to_h(visible, weights, h_bias):
        """
        :param visible:     可见层的值，[1,0,0,0.....]  input_size
        :param weights
        :param h_bias
        :return:            隐含层的概率  [0.1,0.2,...]  input_size
        """
        return tf.sigmoid(tf.matmul(visible, weights) + h_bias)

    def given_v_sample_h(self, visible, weights, h_bias):
        """
        给定可见层的值，通过概率求出隐含层的值
        :param visible:     可见层的值
        :param weights
        :param h_bias
        :return:            隐含层的值
        """
        return self.sample_from_probability(self.probability_v_to_h(visible, weights, h_bias))

    def given_h_sample_v(self, hidden, weights, v_bias):
        """
        给定隐含层的值，通过概率求出可见层的值
        :param hidden:      隐含层
        :param weights
        :param v_bias
        :return:            可见层
        """
        return self.sample_from_probability(self.probability_h_to_v(hidden, weights, v_bias))

    def gibbs_vhv(self, visible, weights, h_bias, v_bias):
        """
        给定可见层，进行一步gibbs采样
        :param visible:
        :param weights
        :param v_bias
        :param h_bias
        :return:
        """
        h_sample = self.given_v_sample_h(visible, weights, v_bias, )
        v_sample = self.given_h_sample_v(h_sample, weights, v_bias)

        return [h_sample, v_sample]

    def gibbs_hvh(self, hidden, weights, v_bias, h_bias):
        """
        给定隐含层，进行一步gibbs采样
        :param hidden:
        :param weights
        :param h_bias
        :param v_bias
        :return:
        """
        v_sample = self.given_h_sample_v(hidden, weights, v_bias)
        h_sample = self.given_v_sample_h(v_sample, weights, h_bias)
        return [v_sample, h_sample]

    def train(self, data):
        _ts_w = tf.placeholder(tf.float32, [self.input_size, self.output_size])
        _ts_vb = tf.placeholder(tf.float32, [self.input_size])
        _ts_hb = tf.placeholder(tf.float32, [self.output_size])

        # _ts_velocity_w = tf.placeholder(tf.float32, [self.input_size, self.output_size])
        # _ts_velocity_vb = tf.placeholder(tf.float32, [self.input_size])
        # _ts_velocity_hb = tf.placeholder(tf.float32, [self.output_size])
        #
        # _current_velocity_w = np.zeros([self.input_size, self._output_size], np.float32)
        # _current_velocity_vb = np.zeros([self.input_size], np.float32)
        # _current_velocity_hb = np.zeros([self.output_size], np.float32)

        _ts_v0 = tf.placeholder(tf.float32, [None, self.input_size])  # 输入的batch
        _ts_probability_h0 = self.probability_v_to_h(_ts_v0, _ts_w, _ts_hb)
        _ts_h0 = self.sample_from_probability(_ts_probability_h0)
        _ts_probability_v1 = self.probability_h_to_v(_ts_h0, _ts_w, _ts_vb)
        _ts_v1 = self.sample_from_probability(_ts_probability_v1)
        _ts_probability_h1 = self.probability_v_to_h(_ts_v1, _ts_w, _ts_hb)
        _ts_h1 = self.sample_from_probability(_ts_probability_h1)

        _gradient_positive = tf.matmul(tf.transpose(_ts_v0), _ts_probability_h0)
        _gradient_negative = tf.matmul(tf.transpose(_ts_v1), _ts_probability_h1)

        update_w = _ts_w + self.params["learning_rate"] * (_gradient_positive - _gradient_negative) / tf.to_float(
            tf.shape(_ts_v0)[0])

        update_vb = _ts_vb + self.params["learning_rate"] * (tf.reduce_mean(_ts_v0 - _ts_v1))

        update_hb = _ts_hb + self.params["learning_rate"] * (tf.reduce_mean(_ts_probability_h0 - _ts_probability_h1))

        start_indexes, end_indexes = self.split_batch(len(data), self.params["batch_size"])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.params["epochs"]):
                for start, end in zip(start_indexes, end_indexes):
                    batch = data[start:end]
                    _current_w = sess.run(update_w,
                                          feed_dict={_ts_w: self.w, _ts_vb: self.vb, _ts_hb: self.hb, _ts_v0: batch})
                    _current_vb = sess.run(update_vb,
                                           feed_dict={_ts_w: self.w, _ts_vb: self.vb, _ts_hb: self.hb, _ts_v0: batch})
                    _current_hb = sess.run(update_hb,
                                           feed_dict={_ts_w: self.w, _ts_vb: self.vb, _ts_hb: self.hb, _ts_v0: batch})
                    self.w = _current_w
                    self.vb = _current_vb
                    self.hb = _current_hb

    def rbm_v_to_h(self, data):
        """
        输入data输出隐含层
        :param data:        np.array([[.....],[......]....])
        :return:           np.array([[.....],[......]....])
        """
        input_x = tf.constant(data, dtype=tf.float32)
        _w = tf.constant(self.w, dtype=tf.float32)
        _hb = tf.constant(self.hb, dtype=tf.float32)
        out = tf.sigmoid(tf.matmul(input_x, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)


class DBN(object):

    def __init__(self, layers, params):
        """
        :param layers:      [] 用于确定每一层的node数量  [visible,h1,h2,h3......]
        :param params:      {} 用于确定epoch，batch
        """
        self.rbms = []
        self.layers = layers
        self.params = params
        for i in range(len(layers) - 1):
            self.rbms.append(RBM(layers[i], layers[i + 1], "h" + str(i), params))

    def train(self, train_data):
        for rbm in self.rbms:
            rbm.train(train_data)
            train_data = rbm.rbm_v_to_h(train_data)
            print(train_data)
        return train_data

    def dbn_up(self, data):
        """
        输入数据到dbn，在隐藏层输出特征
        :param data:
        :return:
        """
        for rbm in self.rbms:
            data = rbm.rbm_v_to_h(data)
        return data


def train_and_test_dbn(project_name, train_name, test_name, dict_params,gv=global_var):
    """
    :param gv: 全局参数
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
    [train_data, test_data] = get_md_data([[train_source_path, train_csv_path], [test_source_path, test_csv_path]],
                                          dict_params['imbalance'])
    _train_ast = train_data.get_ast_vectors()
    _test_ast = test_data.get_ast_vectors()
    from sklearn.preprocessing import minmax_scale
    _train_ast = padding_for_token_batch(_train_ast, train_data.get_vec_len())
    _train_ast = minmax_scale(_train_ast)
    _layers = list()
    _layers.append(len(_train_ast[0]))
    for i in range(dict_params['hidden_layer_num']):
        _layers.append(dict_params['output_size'])

    dbn_name = '%s~~%s' % (train_name, 'dbn')
    dbn = gv.load_dbn(dbn_name)
    if dbn is None:
        dbn = DBN(layers=_layers, params=dict_params)
        logging.info('training dbn .......')
        dbn.train(_train_ast)
        gv.dump_dbn(dbn, dbn_name)

    c_train_x = dbn.dbn_up(_train_ast)
    _train_label = train_data.get_labels()
    _test_label = test_data.get_labels()
    from sklearn.linear_model import LogisticRegression

    cls = LogisticRegression(solver='lbfgs')
    cls.fit(c_train_x, _train_label)
    del _train_ast
    del train_data
    _test_ast = padding_for_token_batch(_test_ast, test_data.get_vec_len())
    _test_ast = minmax_scale(_test_ast)
    c_test_x = dbn.dbn_up(_test_ast)
    _y_predict = cls.predict(c_test_x)
    print_result(y_true=_test_label, y_pred=_y_predict, model='dbn', project_name=project_name,
                 train_name=train_name, test_name=test_name, dict_params=dict_params, sheet_name='dbn')


def train_and_test_dbn_plus(project_name, train_name, test_name, dict_params,gv=global_var):
    """
    :param gv: 全局参数
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
    [train_data, test_data] = get_md_data([[train_source_path, train_csv_path], [test_source_path, test_csv_path]],
                                          dict_params['imbalance'])
    _train_ast = train_data.get_ast_vectors()
    _test_ast = test_data.get_ast_vectors()
    _train_hc = np.array(train_data.get_hand_craft_vectors())
    _test_hc = np.array(test_data.get_hand_craft_vectors())
    from sklearn.preprocessing import minmax_scale
    _train_ast = padding_for_token_batch(_train_ast, train_data.get_vec_len())
    _train_ast = minmax_scale(_train_ast)
    _layers = list()
    _layers.append(len(_train_ast[0]))
    for i in range(dict_params['hidden_layer_num']):
        _layers.append(dict_params['output_size'])

    dbn_name = '%s~~%s' % (train_name, 'dbn_plus')
    dbn = gv.load_dbn(dbn_name)
    if dbn is None:
        dbn = DBN(layers=_layers, params=dict_params)
        logging.info('training dbn plus.......')
        dbn.train(_train_ast)
        gv.dump_dbn(dbn, dbn_name)

    c_train_x = dbn.dbn_up(_train_ast)
    c_train_x = np.hstack((c_train_x, _train_hc))
    _train_label = train_data.get_labels()
    _test_label = test_data.get_labels()
    from sklearn.linear_model import LogisticRegression
    cls = LogisticRegression(solver='lbfgs')
    cls.fit(c_train_x, _train_label)

    del train_data
    del _train_ast
    del _train_hc
    del c_train_x

    _test_ast = padding_for_token_batch(_test_ast, test_data.get_vec_len())
    _test_ast = minmax_scale(_test_ast)
    c_test_x = dbn.dbn_up(_test_ast)
    c_test_x = np.hstack((c_test_x, _test_hc))

    _y_predict = cls.predict(c_test_x)
    print_result(y_true=_test_label, y_pred=_y_predict, model='dbn_plus', project_name=project_name,
                 train_name=train_name, test_name=test_name, dict_params=dict_params, sheet_name='dbn')


if __name__ == '__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    train_and_test_dbn('camel', 'camel-1.2', 'camel-1.4', global_var.dbn_params)
