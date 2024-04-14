import numpy as np
import matplotlib.pyplot as plt
import os
from gensim.models.coherencemodel import CoherenceModel
from seaborn.utils import relative_luminance
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from retrying import retry
from transformers import BertTokenizer
import seaborn as sns
import pandas as pd
import pickle
import logging
import torch
import random


def dir_check(dir_path):
    '''
    dir_path: the path of dir
    '''
    if not os.path.exists(os.path.abspath(dir_path)):
        os.makedirs(os.path.abspath(dir_path))

def pkl_load(file_path):
    '''
    file_path:
    return: obj
    '''
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
        print(" [*] load {}".format(file_path))
        return obj

def pkl_save(file_path, obj):
    try:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
            print(" [*] save {}".format(file_path))
    except Exception as e:
        print(e)   

######################################## logger ########################################
def log_init(log_path, log_name, mode):
    '''
    log_path:   log's dir
    log_name:   log's name
    mode:       w/a
    return:     logger
    '''
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    dir_check(log_path)
    file_path = os.path.join(log_path, (log_name + '.log'))
    formatter = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s - %(process)s')  
    log = logging.FileHandler(file_path, mode=mode, encoding='utf-8')
    log.setFormatter(formatter)
    logger.addHandler(log)
    return logger

######################################## torch ########################################
def seed_init(args):
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    

######################################## array ########################################
def to_np(x):
    '''
    x:  which need to transform numpy
    return: the numpy of x
    '''
    return x.detach().data.cpu().numpy()   


def build_bert_embedding(embedding_fn, vocab, data_dir):
    print(f"building bert embedding matrix for dit {len(vocab)}")
    tokenize = BertTokenizer.from_pretrained('bert-base-uncased')
    embedding_mat_fn = os.path.join(data_dir, f"bert_emb_{len(vocab)}.npy")

    # if matrix exists
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat

    # build bert mat
    index = np.array(
        tokenize.encode(list(vocab.token2id.keys()), add_special_tokens=False))
    bert_mat = np.load(embedding_fn)
    bert_emb = bert_mat[index]
    np.save(embedding_mat_fn, bert_emb)
    return bert_emb


def build_embedding(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix for dict {len(vocab)} if need...")
    dir_check(data_dir)
    embedding_mat_fn = os.path.join(data_dir,
                                    f"embedding_mat_{len(vocab)}.npy")

    # if matrix exists
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat

    # build embedding mat
    embedding_index = {}
    with open(embedding_fn, encoding='UTF-8') as fin:
        first_line = True
        l_id = 0
        for line in fin:
            if l_id % 100000 == 0:
                print("loaded %d words embedding..." % l_id)
            if ("glove" not in embedding_fn) and first_line:
                first_line = False
                continue
            line = line.rstrip()
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            l_id += 1

    embedding_dim = len(list(embedding_index.values())[0])
    embedding_mat = np.zeros(
        (len(vocab) + 1, embedding_dim))  # -1 is for padding
    for i, word in vocab.items():
        embedding_vec = embedding_index.get(word)
        if embedding_vec is not None:
            embedding_mat[i] = embedding_vec
    np.save(embedding_mat_fn, embedding_mat)
    return embedding_mat


def build_embedding_PCA(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix to PCA for dict {len(vocab)} if need...")
    embedding_PCA_fn = os.path.join(data_dir,
                                    f"embedding_mat_PCA_{len(vocab)}.npy")

    if os.path.exists(embedding_PCA_fn):
        embedding_PCA_mat = np.load(embedding_PCA_fn)
        return embedding_PCA_mat

    embedding_mat = build_embedding(embedding_fn, vocab, data_dir)
    pca = PCA(n_components=2)
    embedding_PCA_mat = pca.fit_transform(embedding_mat)
    np.save(embedding_PCA_fn, embedding_PCA_mat)
    return embedding_PCA_mat

def compute_coherence(doc_word, topic_word):
    topic_size, word_size = np.shape(topic_word)
    doc_size = np.shape(doc_word)[0]
    
    coherence = []
    for N in [5, 10, 15]:
        # find top words'index of each topic
        topic_list = []
        for topic_idx in range(topic_size):
            top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)

        # compute coherence of each topic
        sum_coherence_score = 0.0
        for i in range(topic_size):
            word_array = topic_list[i]
            sum_score = 0.0
            for n in range(N):
                flag_n = doc_word[:, word_array[n]] > 0
                p_n = np.sum(flag_n) / doc_size
                for l in range(n + 1, N):
                    flag_l = doc_word[:, word_array[l]] > 0
                    p_l = np.sum(flag_l)
                    p_nl = np.sum(flag_n * flag_l)
                    if p_n * p_l * p_nl > 0:
                        p_l = p_l / doc_size
                        p_nl = p_nl / doc_size
                        sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
            sum_coherence_score += sum_score * (2 / (N * N - N))
        sum_coherence_score = sum_coherence_score / topic_size
        coherence.append(sum_coherence_score)
    return np.mean(coherence)


def evaluate_coherence(topic_words, texts, vocab):
    coherence = {}
    methods = ["c_v", "c_npmi", "c_uci", "u_mass"]
    for method in methods:
        coherence[method] = CoherenceModel(topics=topic_words,
                                           texts=texts,
                                           dictionary=vocab,
                                           coherence=method).get_coherence()
    return coherence


def evaluate_TU(topic_word, n_list=[5,10,15]):
    TU = 0.0
    for n in n_list:
        TU += compute_TU(topic_word, n)
    TU /= len(n_list)
    return TU

def compute_TU(topic_word, N):
    topic_size, word_size = np.shape(topic_word)
    if topic_size == 0:
        return 0
    else:
        topic_list = []
        for topic_idx in range(topic_size):
            top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)
        TU = 0
        cnt = [0 for i in range(word_size)]
        for topic in topic_list:
            for word in topic:
                cnt[word] += 1
        for topic in topic_list:
            TU_t = 0
            for word in topic:
                TU_t += 1 / cnt[word]
            TU_t /= N
            TU += TU_t
        TU /= topic_size
        return TU
    
def evaluate_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words, []))
    total = sum(topic_words, [])
    return len(vocab) / len(total)


def compute_topic_specialization(topic_dist, corpus):
    curpus_vector = corpus.sum(axis=0)
    curpus_vector = curpus_vector / np.linalg.norm(curpus_vector)
    for i in range(topic_dist.shape[0]):
        topic_dist[i] = topic_dist[i] / np.linalg.norm(topic_dist[i])
    topics_spec = 1 - topic_dist.dot(curpus_vector)
    depth_spec = np.mean(topics_spec)
    return depth_spec



def compute_clnpmi(level1, level2, doc_word):

    sum_coherence_score = 0.0
    c = 0

    for N in [5,10,15]:
        word_idx1 = np.argpartition(level1, -N)[-N:]
        word_idx2 = np.argpartition(level2, -N)[-N:]
        
        sum_score = 0.0
        set1 = set(word_idx1)
        set2 = set(word_idx2)
        inter = set1.intersection(set2)
        word_idx1 = list(set1.difference(inter))
        word_idx2 = list(set2.difference(inter))

        for n in range(len(word_idx1)):
            flag_n = doc_word[:, word_idx1[n]] > 0
            p_n = np.sum(flag_n) / len(doc_word)
            for l in range(len(word_idx2)):
                flag_l = doc_word[:, word_idx2[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_nl == len(doc_word):
                    sum_score += 1
                elif p_n * p_l * p_nl > 0:
                    p_l = p_l / len(doc_word)
                    p_nl = p_nl / len(doc_word)
                    p_nl += 1e-10
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
                c += 1
        if c > 0:
            sum_score /= c
        else:
            sum_score = 0
        sum_coherence_score += sum_score
    return sum_coherence_score / 3


def print_topic_word(topic_word, vocab, N):
    topic_size, word_size = np.shape(topic_word)

    top_word_idx = np.argsort(topic_word, axis=1)
    top_word_N = top_word_idx[:, -N:]

    for k, top_word_k in enumerate(top_word_N[:, ::-1]):
        top_words = [vocab[id] for id in top_word_k]
        print(f'Topic {k}:{top_words}')



def evaluate_NPMI(test_data, topic_dist, n_list=[5,10,15]):
    
    return compute_coherence(test_data, topic_dist)

# def build_level(adj_matrix_list: np.array, threshold=0.5):
#     relations = []
#     for idx in range(len(adj_matrix_list)):
#         adj_matrix = np.array(adj_matrix_list[idx])
#            #TODO
#         adj_matrix[adj_matrix < threshold] = 0
#         adj_matrix[adj_matrix > threshold] = 1
        
#         relations.append(adj_matrix)
#     return relations

def build_level(adj_matrix):
    
    adj_matrix[adj_matrix <= 0.025] = 0
    
    result = np.zeros_like(adj_matrix)
    indices = np.expand_dims(np.argmax(adj_matrix, axis=1), axis=1)
    np.put_along_axis(result, indices, 1, axis=1)
    # adj_matrix[adj_matrix < 0.5] = 0
    # adj_matrix[adj_matrix > 0.5] = 1
    for idx, adj in enumerate(adj_matrix):
        if np.max(adj) == 0:
            result[idx] = 0
    
    adj_matrix = result
    topic_num = adj_matrix.shape[0]
    roots = np.where((adj_matrix.sum(axis=1) == 0) | (np.diag(adj_matrix) > 0.5 ))[0]
    adj_matrix = adj_matrix - np.diag(np.diag(adj_matrix))
    
    trees = {}
    # print(roots)
    for root in roots:
        level = {}
        cur = np.zeros(topic_num)
        l = 0
        cur[root] = 1
        while cur.sum() > 0:
            cur_node = np.where(cur > 0)
            level[l] = cur_node[0]
            cur = cur @ adj_matrix.T
            # print(cur.sum(),cur)
            l += 1
        trees[root] = level
    relation = np.where(adj_matrix == 1)
    relation = dict(zip(relation[0], relation[1]))
    return trees, relation

def save_adj_matrix(adj_matrix, epochs, temperature, tables_path):
    dir_check(tables_path)
    adj_df = pd.DataFrame(adj_matrix).stack().reset_index()
    adj_df.columns = ['Topic src', 'Topic dst','Weight']
    adj_df['Epoch'] = epochs
    adj_df['Temperature'] = temperature
    adj_df.to_csv(f'{tables_path}/matrix_{epochs}_{temperature}.csv')

    plt.figure()
    # sns.heatmap(data=adj_matrix,vmax=1.0,vmin=0.0,norm=LogNorm())
    sns.heatmap(data=adj_matrix,norm=LogNorm())

    plt.savefig(f'{tables_path}/matrix_{epochs}_{temperature}.pdf')
    plt.close()

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x