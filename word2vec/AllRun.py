import model.model_with_word_to_vec as md
import model.CNN as pc
import model.DBN as dn
from common.GlobalVariable import GlobalVariable as gv
import common.GlobalVariable as cgv

projects = {'jedit': ['jedit-4.1', 'jedit-4.2', 'jedit-4.3']}


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
    for project_name, sources in cgv.projects.items():
        print('processing project %s' % project_name)
        for i in range(len(sources) - 1):
            print('begin training %s' % sources[i])
            md.train_and_test_cnn(project_name=project_name, train_name=sources[i], test_name=sources[i + 1])
            md.train_and_test_cnn_p(project_name=project_name, train_name=sources[i], test_name=sources[i + 1])


if __name__ == '__main__':
    train_w2v_cnn()
