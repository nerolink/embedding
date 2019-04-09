from common.GlobalVariable import GlobalVariable as gv
import os


def delete(project_name):
    if os.path.exists(os.path.join(gv.token_len_path, '%s.tvl' % project_name)):
        os.remove(os.path.join(gv.token_len_path, '%s.tvl' % project_name))
    if os.path.exists(os.path.join(gv.hf_tree_path, '%s.ht' % project_name)):
        os.remove(os.path.join(gv.hf_tree_path, '%s.ht' % project_name))
    if os.path.exists(os.path.join(gv.word_to_vec_path, '%s_%d.w2v' % (project_name, gv.w2v_cnn_params['vec_size']))):
        os.remove(os.path.join(gv.word_to_vec_path, '%s_%d.w2v' % (project_name, gv.w2v_cnn_params['vec_size'])))
    if os.path.exists(os.path.join(gv.word_to_vec_path, '%s_%d.txt' % (project_name, gv.w2v_cnn_params['vec_size']))):
        os.remove(os.path.join(gv.word_to_vec_path, '%s_%d.txt' % (project_name, gv.w2v_cnn_params['vec_size'])))


if __name__ == '__main__':
    delete('xalan')
