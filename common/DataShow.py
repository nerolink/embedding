import pandas as pd
from GlobalVariable import GlobalVariable as gv
import os


def watch_result():
    df = pd.read_csv(os.path.join(gv.result_path, 'cnn_w2v.csv'))
    df['f1-score'].groupby(df['train_project']).max().to_csv('cnn_w2v', index=True)


def watch_result_plain():
    df = pd.read_csv(os.path.join(gv.result_path, 'cnn_plain.csv'))
    df['f1-score'].groupby(df['train_project']).max().to_csv('cnn_plain', index=True)


if __name__ == '__main__':
    watch_result_plain()
    watch_result()
