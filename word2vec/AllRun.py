import model.model_with_word_to_vec as md
import model.CNN as pc
import model.DBN as dn
from common.GlobalVariable import GlobalVariable as gv
import common.GlobalVariable as cgv
from common.GlobalVariable import candidate
import logging
import time

projects = {'jedit': ['jedit-4.1', 'jedit-4.2', 'jedit-4.3']}
logging.basicConfig(format=gv.config['logging_format'], level=gv.config['logging_level'])


def train_test():
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


def train_all():
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


def train_w2v_cnn():
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


def train_plain_cnn():
    for project_name, sources in cgv.projects.items():
        for i in range(len(sources) - 1):
            logging.critical('begin processing %s ' % sources[i])
            start = time.time()
            pc.train_and_test_cnn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                  dict_params=gv.plain_cnn_params)
            pc.train_and_test_cnn_p(project_name=project_name, train_name=sources[i], test_name=sources[i + 1],
                                    dict_params=gv.plain_cnn_params)
            logging.critical('finish in %d seconds' % (time.time() - start))


def train_dbn():
    for project_name, sources in cgv.projects.items():
        for i in range(len(sources) - 1):
            logging.critical('begin processing %s ' % sources[i])
            start = time.time()
            dn.train_and_test_dbn(project_name, sources[i], sources[i + 1], gv.dbn_params)
            dn.train_and_test_dbn_plus(project_name, sources[i], sources[i + 1], gv.dbn_params)


if __name__ == '__main__':
    train_w2v_cnn()
    # train_plain_cnn()
    # train_dbn()
