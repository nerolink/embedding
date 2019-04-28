import os

import model.model_with_word_to_vec as md
import model.CNN as pc
import model.DBN as dn
from common.GlobalVariable import instance as global_var
import common.GlobalVariable as cgv
from common.GlobalVariable import candidate
import logging
import time
import HierarchicalSoftmax as hs
from model.LR import train_and_test_lr
import threading

projects = {'jedit': ['jedit-4.1', 'jedit-4.2', 'jedit-4.3']}
logging.basicConfig(format=global_var.config['logging_format'], level=global_var.config['logging_level'])


def train_test(gv=global_var):
    for project_name, sources in projects.items():
        for i in range(len(sources) - 1):
            print('begin training %s' % sources[i])
            # md.train_and_test_cnn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1])
            # md.train_and_test_cnn_p(project_name=project_name, train_name=sources[i], test_name=sources[i + 1])
            pc.train_and_test_cnn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                  dict_params=gv.plain_cnn_params)
            pc.train_and_test_cnn_p(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                    dict_params=gv.plain_cnn_params)
            # dn.train_and_test_dbn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
            #                       dict_params=gv.dbn_params)
            # dn.train_and_test_dbn_plus(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
            #                            dict_params=gv.dbn_params)


def train_all(gv=global_var):
    for project_name, sources in cgv.projects.items():
        for i in range(len(sources) - 1):
            print('begin training %s' % sources[i])
            md.train_and_test_cnn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1])
            md.train_and_test_cnn_p(project_name=project_name, train_name=sources[i], test_name=sources[i + 1])
            pc.train_and_test_cnn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                  dict_params=gv.plain_cnn_params)
            pc.train_and_test_cnn_p(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                    dict_params=gv.plain_cnn_params)
            dn.train_and_test_dbn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                  dict_params=gv.dbn_params)
            dn.train_and_test_dbn_plus(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                       dict_params=gv.dbn_params)


def train_w2v_cnn(gv=global_var):
    for vec_size in candidate['vec_size']:
        gv.w2v_cnn_params['vec_size'] = vec_size
        logging.critical('begin vec_size = %d' % vec_size)
        for project_name, sources in cgv.projects.items():
            for i in range(len(sources) - 1):
                logging.critical('begin processing %s ' % sources[i])
                start = time.time()
                md.train_and_test_cnn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1])
                md.train_and_test_cnn_p(project_name=project_name, train_name=sources[i], test_name=sources[i + 1])
                logging.critical('finish in %d seconds' % (time.time() - start))


def train_plain_cnn(gv=global_var):
    for project_name, sources in cgv.projects.items():
        for i in range(len(sources) - 1):
            logging.critical('begin processing %s ' % sources[i])
            start = time.time()
            pc.train_and_test_cnn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                  dict_params=gv.plain_cnn_params)
            pc.train_and_test_cnn_p(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                    dict_params=gv.plain_cnn_params)
            logging.critical('finish in %d seconds' % (time.time() - start))


def train_dbn(gv=global_var):
    for project_name, sources in cgv.projects.items():
        for i in range(len(sources) - 1):
            logging.critical('begin processing %s ' % sources[i])
            start = time.time()
            dn.train_and_test_dbn(project_name, sources[i], sources[i + 1], gv.dbn_params)
            dn.train_and_test_dbn_plus(project_name, sources[i], sources[i + 1], gv.dbn_params)


class TaskTrainW2v(threading.Thread):
    def __init__(self, project_name, vec_size) -> None:
        super().__init__()
        self.start = time.time()
        self.project_name = project_name
        self.vec_size = vec_size

    def run(self) -> None:
        import time
        from GlobalVariable import GlobalVariable
        print('start training project:%s,vec_size %d:' % (self.project_name, self.vec_size))
        g_v = GlobalVariable()
        g_v.w2v_cnn_params['vec_size'] = self.vec_size
        hs.train(self.project_name, g_v)
        print('finish training project:%s,vec_size %d:,cost:%s'
              % (self.project_name, self.vec_size, time.time() - self.start))


def train_w2v():
    start = time.time()
    tasks = []
    for vec_size in candidate['vec_size']:
        for project_name in cgv.projects.keys():
            tasks.append(TaskTrainW2v(project_name, vec_size))
            tasks[-1].start()
    for task in tasks:
        task.join()
    print('cost %d seconds' % (time.time() - start))


def train_lr():
    for project_name, sources in cgv.projects.items():
        for i in range(len(sources) - 1):
            train_and_test_lr(project_name, sources[i], sources[i + 1], {})


def del_model():
    import os
    for root, dirs, files in os.walk(global_var.data_path + 'model'):
        for file in files:
            if file.__contains__('plain') or file.__contains__('dbn'):
                os.remove(os.path.join(root, file))


if __name__ == '__main__':
    # from keras.models import load_model
    #
    # for root, dirs, files in os.walk(global_var.data_path + '/model'):
    #     for file in files:
    #         if not file.__contains__('cnn'):
    #             continue
    #         print(file)
    #         load_model(os.path.join(root, file))

    # train_plain_cnn()
    # train_dbn()
    train_w2v_cnn()
