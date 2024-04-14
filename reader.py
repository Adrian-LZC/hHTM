import os
import utils
import warnings
import numpy as np
from typing import Counter
import gensim
from collections import Counter, defaultdict
from numpy.core.fromnumeric import shape
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.preprocessing import normalize as sknormalize
from sklearn.feature_extraction.text import TfidfTransformer

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

use_vocab_datasets = {"20news_1995"}

filter_map = {
    "20news": True,
    "20news_1995": False,

    "rcv1": False,
    "wikitext-103": False,
    "reuters": False,
    "TMN": False,

    "DBpedia": True,
    "AGnews": True,
    "R21578": True,
    "nips": True,

    "nyt": True,
    "arxiv":True,
    "classic4":False,
    'yelp':True
}

def dump_vocab(path, vocab):
    with open(path, "w") as fout:
        for id in vocab.keys():
            fout.write(f"{vocab[id]} 0\n")
            
class TextReader:
    def __init__(self, dataset_name, data_path):
        data_path = f"{data_path}/{dataset_name}"
        utils.dir_check(f"{data_path}/mat")
        utils.dir_check(f"{data_path}/feat")
        utils.dir_check(f"{data_path}/embs")
        utils.dir_check(f"{data_path}/processed")

        self.processed_path = f"{data_path}/processed"
        self.feat_path = f"{data_path}/feat"
        self.embs_path = f"{data_path}/embs"
        self.mat_path = f"{data_path}/mat"

        print("Dataset preparing....")
        self.dataset_name = dataset_name
        self.base_path = data_path  # 20news
        self.vocab_path = os.path.join(
            self.base_path, "processed/vocab.pkl"
        )  # 20news/processed/vocab.pkl

        self.train_cls = [] #utils.pkl_load(os.path.join(cls_path, dataset_name, 'train_cls.pkl'))
        self.valid_cls = [] #utils.pkl_load(os.path.join(cls_path, dataset_name, 'valid_cls.pkl'))
        self.test_cls = [] #utils.pkl_load(os.path.join(cls_path, dataset_name, 'test_cls.pkl'))
        
        # load or process
        self._rebuild_vocab()
        # try:
        #     self._load()
        # except:
        #     print("rebuilding vocab ...")

        # read all txt

        self.idx2word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"Load {self.vocab_size} words..")

    def _load(self):
        self._read_data()

        processed_train_path = os.path.join(
            self.base_path, "processed/train_data.pkl")
        processed_valid_path = os.path.join(
            self.base_path, "processed/valid_data.pkl")
        processed_test_path = os.path.join(
            self.base_path, "processed/test_data.pkl")

        self.vocab = utils.pkl_load(self.vocab_path)
        self.train_data = utils.pkl_load(processed_train_path)
        self.train_label = utils.pkl_load(processed_train_path.replace(
            "_data.pkl", "_label.pkl")).astype(np.int16)

        self.valid_data = utils.pkl_load(processed_valid_path)
        self.valid_label = utils.pkl_load(processed_valid_path.replace(
            "_data.pkl", "_label.pkl")).astype(np.int16)

        self.test_data = utils.pkl_load(processed_test_path)
        self.test_label = utils.pkl_load(processed_test_path.replace(
            "_data.pkl", "_label.pkl")).astype(np.int16)
    # data: label \t text
    def _read_text(self, file_path):
        labels = []
        texts = []
        cls = []
        with open(file_path, encoding="utf-8") as fin:
            lines = fin.read().split("\n")
            for n_i, line in enumerate(lines):
                if not line:
                    continue

                label, text = line.strip().split("\t", 1)
                text = list(gensim.utils.tokenize(text, lower=True))
                texts.append(text)
                labels.append(label)

            return labels, texts

    def _build_vocab(self, frequence):
        self.label_dict = {}
        try:
            with open(os.path.join(self.base_path, "raw/labels.txt"), "r") as f:
                f = f.read()
                l = f.split("\n")
                for id, label in enumerate(l):
                    self.label_dict[label.lower()] = id
        except:
            pass
        self._read_data()

        all_text = self.train_text + self.valid_text + self.test_text

        # build dictionary
        dictionary = gensim.corpora.Dictionary(all_text)
        print("Before", len(dictionary))
        dictionary = self._filter_(dictionary, frequence)
        print("After", len(dictionary))
        self.vocab = dictionary

        # 保存词汇表
        print("utils.pkl_save ...")
        utils.pkl_save(self.vocab_path, self.vocab)
        utils.pkl_save(f'{self.base_path}/raw/vocab.pkl', self.vocab)

        dump_vocab(
            f"{self.base_path}/feat/{self.dataset_name}.vocab", self.vocab)
        if self.dataset_name not in use_vocab_datasets:
            dump_vocab(
                f"{self.base_path}/raw/{self.dataset_name}.vocab", self.vocab)
            
    def _read_data(self):
        train_path = os.path.join(self.base_path, "raw/train.txt")
        valid_path = os.path.join(self.base_path, "raw/valid.txt")
        test_path = os.path.join(self.base_path, "raw/test.txt")

        self.train_label, self.train_text = self._read_text(train_path)
        print(f"train_label: {len(self.train_label)}, train_text: {len(self.train_text)}")
        self.valid_label, self.valid_text = self._read_text(valid_path)
        print(f"valid_label: {len(self.valid_label)}, valid_text: {len(self.valid_text)}")
        self.test_label, self.test_text = self._read_text(test_path)
        print(f"test_label: {len(self.test_label)}, test_text: {len(self.test_text)}")

    def _read_processed_data(self):
        pass
        
    def _filter_(self, dictionary, frequence):
        if self.dataset_name in use_vocab_datasets:
            print(
                f"{self.dataset_name} loads vocab from {self.base_path}/raw/{self.dataset_name}.vocab")
            words = []
            with open(f'{self.base_path}/raw/{self.dataset_name}.vocab', "r") as fin:
                for line in fin.readlines():
                    words.append(line.split(" ")[0])
            dictionary.filter_tokens(good_ids=list(
                map(dictionary.token2id.get, words)))
        elif filter_map[self.dataset_name]:
            print(f"{self.dataset_name} filters vocab ...")
            dictionary.filter_tokens(
                list(map(dictionary.token2id.get, STOPWORDS)))
            len_1_words = list(
                filter(lambda w: len(w) < 2, dictionary.values()))
            dictionary.filter_tokens(
                list(map(dictionary.token2id.get, len_1_words)))
            dictionary.filter_extremes(no_below=frequence)
        else:
            print(f"{self.dataset_name} use all words")

        dictionary.compactify()
        return dictionary

    def _rebuild_vocab(self, frequence=50):
        train_path = os.path.join(self.base_path, "raw/train.txt")
        valid_path = os.path.join(self.base_path, "raw/valid.txt")
        test_path = os.path.join(self.base_path, "raw/test.txt")

        self._build_vocab(frequence)
        #TODO
        self.train_data, self.train_label, self.train_text, self.train_cls = self._file_to_data(
            train_path, self.train_cls
        )
        self.valid_data, self.valid_label, self.valid_text, self.valid_cls = self._file_to_data(
            valid_path, self.valid_cls
        )
        self.test_data, self.test_label, self.test_text, self.test_cls = self._file_to_data(
            test_path, self.test_cls)
        
        

    def _read_data(self):
        train_path = os.path.join(self.base_path, "raw/train.txt")
        valid_path = os.path.join(self.base_path, "raw/valid.txt")
        test_path = os.path.join(self.base_path, "raw/test.txt")

        self.train_label, self.train_text = self._read_text(train_path)
        self.valid_label, self.valid_text = self._read_text(valid_path)
        self.test_label, self.test_text = self._read_text(test_path)

    def _file_to_data(self, file_path, bert_cls):
        labels, texts = self._read_text(file_path)

        data = []
        m_labels = []
        new_texts = []
        new_cls = []
        for label, text in zip(labels, texts):
            if len(self.vocab.doc2bow(text)) < 3:
                continue
            new_texts.append(text)
            # new_cls.append(b_cls)
            word = list(map(self.vocab.token2id.get, text))
            word = np.array(
                list(filter(lambda x: x is not None, word)), dtype=object)

            if label.isdigit():
                m_labels.append(int(label))
            elif ":" in label:
                l = label.split(":")[0]
                m_labels.append(self.label_dict[l.lower()])
            else:
                m_labels.append(self.label_dict[label.lower()])
            data.append(word)

        lens = list(map(len, data))
        print(
            " [*] load {} docs, avg len: {}, max len: {}".format(
                len(data), np.mean(lens), np.max(lens)
            )
        )

        data = np.array(data, dtype=object)
        utils.pkl_save(
            file_path.replace(
                "/raw/", "/processed/").replace(".txt", "_data.pkl"), data
        )

        m_labels = np.array(m_labels, dtype=object)
        if m_labels.max() == self.get_n_classes():
            m_labels = m_labels - 1
        utils.pkl_save(
            file_path.replace(
                "/raw/", "/processed/").replace(".txt", "_label.pkl"),
            m_labels,
        )
        return data, m_labels, new_texts, new_cls

    def get_sequence(self, data_type):
        if data_type == "train":
            return (self.train_data, self.train_label, self.train_text)

        elif data_type == "valid":
            return (self.valid_data, self.valid_label, self.valid_text)

        elif data_type == "test":
            return (self.test_data, self.test_label, self.test_text)

        elif data_type == "train+valid":
            data = np.concatenate([self.train_data, self.valid_data], axis=0)
            label = np.concatenate([self.train_label, self.valid_label], axis=0)
            # text = np.concatenate([self.train_text, self.valid_text], axis=0)
            text = self.train_text + self.valid_text
            return (data, label, text)

        elif data_type == "all":
            data = np.concatenate(
                [self.train_data, self.valid_data, self.test_data])
            label = np.concatenate(
                [self.train_label, self.valid_label, self.test_label]
            )
            # text = np.concatenate(
            #     [self.train_text, self.valid_text, self.test_text])
            text = self.train_text + self.valid_text + self.test_text
            return (data, label, text)
        else:
            raise Exception(" [!] Unkown data type : {}".format(data_type))
        
    def get_n_classes(self):
        if hasattr(self, "n_classes"):
            return self.n_classes
        else:
            with open(self.base_path + '/raw/labels.txt', 'r') as f:
                f = f.read()
                l = f.split('\n')
                self.n_classes = len(l)
                return self.n_classes  
           
    def get_bow(self, data_type="train"):
        raw_data, raw_label, raw_text = self.get_sequence(data_type)

        bow = []
        for item in raw_data:
            bow.append(list(Counter(item).items()))
        return bow
    
    def get_matrix(self, data_type="train", mode="onehot", normalize=False):
        raw_data, raw_label, raw_text = self.get_sequence(data_type)

        matrix_path = os.path.join(
            self.base_path, f"processed/matrix_{data_type}_{mode}_{normalize}.pkl"
        )

        if os.path.exists(matrix_path):
            x = utils.pkl_load(matrix_path)
            return [x, raw_label, raw_text]

        x = np.zeros((len(raw_data), self.vocab_size))
        for i, seq in enumerate(raw_data):
            counter = defaultdict(int)
            for j in seq:
                counter[j] += 1

            total_num = sum(counter.values())
            for j, c in list(counter.items()):
                if mode == "count":
                    x[i][j] = c
                elif mode == "freq":
                    x[i][j] = c / len(seq)
                elif mode == "onehot":
                    x[i][j] = 1
                elif mode == "tfidf":
                    tf = 1 + np.log(c)
                    idf = np.log(1 + self.vocab.num_docs / 
                                 (1 + self.vocab.dfs[j]))
                    x[i][j] = tf * idf
                else:
                    raise ValueError("Unknown vectorization mode:", mode)

        if normalize:
            x = sknormalize(x, norm="l1", axis=1)
            x = x.astype(np.float32)

        return [x, raw_label, raw_text]

    def get_tfidf(self, data_type='train'):
        bow, label, text = self.get_matrix(data_type, mode='count')
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(bow)
        return tfidf.toarray(), label, text
        
    
    def generator_matrix(
        self,
        data_type="train",
        batch_size=32,
        rand=True,
        mode="onehot",
        normalize=False,
    ):
        raw_data, raw_label, raw_text = self.get_sequence(data_type)

        count = 0
        while True:
            if not rand:
                beg = (count * batch_size) % raw_data.shape[0]
                end = ((count + 1) * batch_size) % raw_data.shape[0]
                if beg > end:
                    beg -= raw_data.shape[0]

                idx = np.arange(beg, end)
            else:
                idx = np.random.randint(0, len(raw_data), batch_size)

            data = raw_data[idx]
            label = raw_label[idx]
            text = raw_text[idx]

            x = np.zeros((len(data), self.vocab_size))
            for i, seq in enumerate(data):
                counter = defaultdict(int)
                for j in seq:
                    counter[j] += 1

                total_num = sum(counter.values())
                for j, c in list(counter.items()):
                    if mode == "count":
                        x[i][j] = c
                    elif mode == "freq":
                        x[i][j] = c / len(seq)
                    elif mode == "onehot":
                        x[i][j] = 1
                    elif mode == "tfidf":
                        tf = 1 + np.log(c)
                        idf = np.log(1 + self.vocab.num_docs / 
                                        (1 + self.vocab.dfs[j]))
                        x[i][j] = tf * idf
                    else:
                        raise ValueError("Unknown vectorization mode:", mode)

            if normalize:
                x = sknormalize(x, norm="l2", axis=1)
                x = x.astype(np.float32)

            yield [x, label, text]

            count += 1
    