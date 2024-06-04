# 构建图
import numpy as np
from numpy.linalg import norm
from scipy import sparse
import math
from typing import List, Dict, Set
import pickle
import os


def read_all_documents():
    return 


def get_windows(document_list: np.ndarray, window_size=20):
    print("Split windows...")
    windows: List[np.ndarray] = []
    for idx, document_words in enumerate(document_list):
        print("get windows, index {}".format(idx))
        length = len(document_words) # 500
        if length <= window_size:
            windows.append(document_words)
        else:
            for j in range(length - window_size + 1):
                window = document_words[j: j + window_size]
                windows.append(window)
    return windows


def statistic_word_freq(windows: List):

    word_window_freq: Dict[int, int] = {}
    for idx, window in enumerate(windows):
        print('Coping window {}'.format(idx))
        appeared: Set[int] = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])
    return word_window_freq


def statistic_pair_freq(windows: List):

    word_pair_count: Dict[str, int] = {}

    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i_id = window[i]
                word_j_id = window[j]
                if word_i_id == word_j_id:
                    # w_i == w_j
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    print("Finish word pairs statistic.")
    return word_pair_count


def fill_word_word_edge(word_window_freq, pair_window_freq, window_nums, vocab_size, thresh=0):
    row = []
    col = []
    weight = []

    for key in pair_window_freq:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        if i >= vocab_size or j >= vocab_size:
            continue
        count = pair_window_freq[key]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = math.log((1.0 * count / window_nums) / (1.0 * word_freq_i * word_freq_j/(window_nums * window_nums)))
        if pmi <= thresh:
            continue
        # TODO A(i, j) = PMI(i, j), if node_i and node_j are both word
        row.append(i)
        col.append(j)
        weight.append(pmi)
    print("Finish constructing edges between word node and word node done(PMI)!")
    return row, col, weight


def fill_word_doc_edge(document_list, word_nums):
    row = []
    col = []
    weight = []
    word_doc_dict: Dict[int, List[int]] = {}

    for i in range(len(document_list)):
        doc_words = document_list[i]
        appeared: Set[int] = set()
        for word_idx in doc_words:
            if word_idx in appeared:
                continue
            if word_idx in word_doc_dict:
                doc_list = word_doc_dict[word_idx]
                doc_list.append(i)
                word_doc_dict[word_idx] = doc_list
            else:
                word_doc_dict[word_idx] = [i]
            appeared.add(word_idx)

    doc_word_count_dict: List[str, int] = {}

    for doc_idx in range(len(document_list)):
        doc_words: np.ndarray = document_list[doc_idx]
        for word_idx in doc_words:
            doc_word_str = str(doc_idx) + ',' + str(word_idx)
            if doc_word_str in doc_word_count_dict:
                doc_word_count_dict[doc_word_str] += 1
            else:
                doc_word_count_dict[doc_word_str] = 1

    for i in range(len(document_list)):
        doc_words = document_list[i]
        doc_word_set = set()
        for word_idx in doc_words:
            if word_idx >= word_nums:
                continue
            if word_idx in doc_word_set:
                continue
            key = str(i) + ',' + str(word_idx)
            count = doc_word_count_dict[key]
            row.append(word_nums + i)
            col.append(word_idx)
            idf = math.log(1.0 * len(document_list) / (len(word_doc_dict[word_idx]) + 1))
            tf = count / len(doc_words)
            tf_idf = tf * idf
            weight.append(tf_idf)
            row.append(word_idx)
            col.append(word_nums + i)
            weight.append(tf_idf)
            doc_word_set.add(word_idx)
    print("Finish constructing edges between word node and document node done(TF-IDF)!")
    return row, col, weight


def get_doc_embeddings(vocab_embeddings: np.ndarray, document_list):
    doc_embeddings = []
    for doc_idx in range(len(document_list)):
        document: np.ndarray = document_list[doc_idx] # 单篇文档
        doc_embeddings.append(vocab_embeddings[document].sum(axis=0) / len(document))
    return doc_embeddings


def get_cosine_similarity(a: np.ndarray, b: np.ndarray):
    cosine = np.dot(a, b) / (norm(a) * norm(b))
    return cosine


def fill_doc_label_edge(doc_embeddings, label_embeddings, word_nums, thresh=0):
    row = []
    col = []
    weight = []
    doc_nums = len(doc_embeddings)
    label_nums = len(label_embeddings)
    for doc_idx in range(doc_nums):
        for label_idx in range(label_nums):
            cosine_similarity = get_cosine_similarity(doc_embeddings[doc_idx], label_embeddings[label_idx])
            if cosine_similarity > thresh:
                row.append(doc_idx + word_nums)
                col.append(label_idx + word_nums + doc_nums)
                weight.append(cosine_similarity)
                row.append(label_idx + word_nums + doc_nums)
                col.append(doc_idx + word_nums)
                weight.append(cosine_similarity)
    return row, col, weight


def fill_word_label_edge(vocab_embeddings, label_embeddings, word_nums, doc_nums, thresh=0):
    row = []
    col = []
    weight = []
    label_nums = len(label_embeddings)
    for word_idx in range(word_nums):
        for label_idx in range(label_nums):
            cosine_similarity = get_cosine_similarity(label_embeddings[label_idx], vocab_embeddings[word_idx])
            if cosine_similarity > thresh:
                row.append(word_idx)
                col.append(label_idx + word_nums + doc_nums)
                weight.append(cosine_similarity)
                row.append(label_idx + word_nums + doc_nums)
                col.append(word_idx)
                weight.append(cosine_similarity)
    return row, col, weight


def fill_main_diagonal(node_nums):
    row = []
    col = []
    weight = []
    for idx in range(node_nums):
        row.append(idx)
        col.append(idx)
        weight.append(1)
    return row, col, weight


def build_G(data_root, x_file, y_file, word_emb_file, label_emb_file, vocab_size, document_nums, label_nums, embed_dim, window_size=20):
    data_x_path = os.path.join(data_root, x_file) # './dataset/aapd/data_X.npy'
    data_X = np.load(data_x_path)
    node_nums = vocab_size + document_nums + label_nums
    print("total documents shape: {}".format(data_X.shape))
    windows = get_windows(data_X, window_size)
    word_window_freq = statistic_word_freq(windows)
    pair_window_freq = statistic_pair_freq(windows)
    w_w_row, w_w_col, w_w_weight = fill_word_word_edge(word_window_freq, pair_window_freq, len(windows), vocab_size=vocab_size)
    w_d_row, w_d_col, w_d_weight = fill_word_doc_edge(data_X, word_nums=vocab_size)
    vocab_embeddings = np.loadtxt(os.path.join(data_root, word_emb_file), delimiter=' ', usecols=(range(1, embed_dim+1)))
    label_embeddings = np.load(os.path.join(data_root, label_emb_file))
    doc_embeddings = get_doc_embeddings(vocab_embeddings, data_X)
    d_l_row, d_l_col, d_l_weight = fill_doc_label_edge(doc_embeddings, label_embeddings, vocab_size)
    w_l_row, w_l_col, w_l_weight = fill_word_label_edge(vocab_embeddings, label_embeddings, vocab_size, doc_nums=len(data_X))
    diagonal_row, diagonal_col, diagonal_weight = fill_main_diagonal(node_nums)
    weight = w_w_weight + w_d_weight + d_l_weight + w_l_weight
    row = w_w_row + w_d_row + d_l_row + w_l_row
    col = w_w_col + w_d_col + d_l_col + w_l_col
    adj = sparse.coo_matrix((weight, (row, col)), shape=(node_nums, node_nums))
    with open(os.path.join(data_root, 'graph.adj'), 'wb') as f:
        pickle.dump(adj, f)
    print(
        "Finished constructing graph including word nodes, document nodes and label nodes(edges between word nodes and label nodes, document nodes and label nodes)!")


if __name__ == '__main__':
    train_size = 53840
    valid_size = 1000
    test_size = 1000
    doc_nums = train_size + valid_size + test_size
    word_nums = 69397
    label_nums = 54
    embed_dim = 300
    build_G(data_root='./dataset/aapd/', x_file='X_data.npy', y_file='y_data.npy', word_emb_file='word_embed.txt', label_emb_file='label_embed.npy', vocab_size=word_nums, document_nums=doc_nums, label_nums=label_nums, embed_dim=embed_dim)