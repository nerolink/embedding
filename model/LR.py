from common.utils import get_md_data, z_score, print_result
from common.GlobalVariable import instance as global_var
import numpy as np
from sklearn.linear_model import LogisticRegression


def train_and_test_lr(project_name, train_name, test_name, dict_params, gv=global_var):
    train_data, test_data = get_md_data(
        [['%s/%s' % (gv.projects_source_dir, train_name), '%s/%s.csv' % (gv.csv_dir, train_name)]
            , ['%s/%s' % (gv.projects_source_dir, test_name), '%s/%s.csv' % (gv.csv_dir, test_name)]])
    train_hand = z_score(np.array(train_data.get_hand_craft_vectors()))
    test_hand = z_score(np.array(test_data.get_hand_craft_vectors()))
    train_y = np.array(train_data.get_labels())
    test_y = np.array(test_data.get_labels())
    cls = LogisticRegression(solver='lbfgs', max_iter=1000)
    cls.fit(train_hand, train_y)
    y_predict = cls.predict(test_hand)
    print_result(y_pred=y_predict, y_true=test_y, sheet_name='LogisticRegression', project_name=project_name,
                 train_name=train_name, test_name=test_name, dict_params=dict_params, model='LR')
