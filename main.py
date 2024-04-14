import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import utils
from reader import TextReader
from torch.distributions import Dirichlet
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset

from gan import Encoder, Generator, Discriminator

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cuda = torch.cuda.is_available()
device = torch.device("cuda")

parser = argparse.ArgumentParser(description="hHTM")

parser.add_argument('--dataset', type=str, default='nips', help="dataset_name")
parser.add_argument('--result_dir', type=str, default='./res', help='the dir of result')
parser.add_argument('--data_dir', type=str, default='./dataset', help='the dir of dataset')
parser.add_argument('--vocab_size', type=int, default=3531, help='vocab size')
parser.add_argument('--emb_type', type=str, default='glove', help='the type of word embedding')
parser.add_argument('--emb_size', type=int, default=300, help='the dim of embedding')
parser.add_argument('--cuda', type=bool, default=True, help='whether use cuda or not')
parser.add_argument('--mode', type=str, default='Train', help="Train, Test, Downstream")
parser.add_argument('--seed', type=int, default=0)


# GMM
parser.add_argument('--model_path', type=str, default='./res/model', help='the dir of model')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=30000)
parser.add_argument('--optim', type=str, default='rmsprop')
parser.add_argument('--weight_decay', default=0.009, type=float, metavar='W', help='weight decay(default: 1e-4)')
parser.add_argument('--topic_num', type=int, default=200, help='the topic num of all layers')

parser.add_argument('--rho_max', type=float, default=1.e+10)
parser.add_argument('--rho', type=float, default=1.0)
parser.add_argument('--phi', type=float, default=1.e+2)
parser.add_argument('--epsilon', type=float, default=0.0)
parser.add_argument('--lam', type=float, default=0.0)

# Encoder
parser.add_argument('--dropout', type=float, default=0.1)

args = parser.parse_args()

class MyDataset(Dataset):
    def __init__(self, x) -> None:
        super().__init__()
        self.x  = x
    
    def __getitem__(self, index):
        return self.x[index]
    
    def __len__(self):
        return len(self.x)
    
# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())
    
class GMM(object):
    def __init__(self,
                 args=None,
                 reader=None,
                 emb_mat=None,
                 logger=None,
                 **kwargs):
        # prepare dataset
        if reader == None:
            raise Exception(" [!] Expected data reader")
        self.reader = reader
        self.train_data, self.train_label, self.train_text = self.reader.get_matrix(
            'train+valid', mode='tfidf', normalize=False)
        self.test_data, self.test_label, self.test_text = self.reader.get_matrix(
            'test', mode='tfidf', normalize=False)
        
        self.train_data = torch.from_numpy(self.train_data).to(torch.float)
        

        self.logger = logger
        self.model_path = os.path.join(args.model_path, args.dataset + '_' + str(args.topic_num))
        
        print("GMM init model.")
        if emb_mat is None:
            self.E = Encoder(args=args, device=device, **kwargs).to(device)
            self.G = Generator(args=args, device=device, **kwargs).to(device)
            self.D = Discriminator(args=args, **kwargs).to(device)
        else:
            emb_mat = torch.from_numpy(emb_mat.astype(np.float32)).to(device)
            self.E = Encoder(args=args, device=device, **kwargs).to(device)
            self.G = Generator(args=args, emb_mat=emb_mat, device=device, **kwargs).to(device)
            self.D = Discriminator(args=args, **kwargs).to(device)
        
        
        self.topic_num = args.topic_num
        self.rho_max = args.rho_max
        self.rho = args.rho
        self.phi = args.phi
        self.epsilon = args.epsilon
        self.lam = args.lam
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.dirichlet = Dirichlet(torch.tensor([50 / self.topic_num] * self.topic_num))
        self.e_optimizer = optim.AdamW(self.E.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        
        if args.optim == 'rmsprop':
            self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=2 * args.learning_rate)
            self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=args.learning_rate)
        elif args.optim == 'adamW':
            self.d_optimizer = optim.AdamW(self.D.parameters(), lr=2 * args.learning_rate, weight_decay=args.weight_decay)
            self.g_optimizer = optim.AdamW(self.G.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        
        self.ce = nn.CrossEntropyLoss()
    
    def fake_loss(self, noise):
        return self.D(self.G(noise), noise)

    def real_loss(self, docs, encoder_output):
        return self.D(docs, encoder_output)

    def adj_loss(self):
        adj_matrix, R = self.G.get_adjacency_matrix()
        d = adj_matrix.shape[0]
        h = torch.trace(torch.matrix_exp(adj_matrix * adj_matrix)) - d  # (Zheng et al. 2018)

        
        f = torch.square(R.norm()) 
        return h, f
    
    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.E.state_dict(), f'{self.model_path}/encoder.pkl')
        torch.save(self.G.state_dict(), f'{self.model_path}/generator.pkl')
        torch.save(self.D.state_dict(), f'{self.model_path}/discriminator.pkl')
        torch.save(self.G.topic_embed, f'{self.model_path}/topic_emb.pkl')
        torch.save(self.G.word_embed, f'{self.model_path}/word_emb.pkl')
        torch.save(self.G.adjacency_matrix, f'{self.model_path}/adj_matrix.pkl')
        print(f'Models save to  {self.model_path}/model.pkl')  
    
    def load_model(self):
        self.E.load_state_dict(torch.load(f'{self.model_path}/encoder.pkl'))
        self.G.load_state_dict(torch.load(f'{self.model_path}/generator.pkl'))
        self.D.load_state_dict(torch.load(f'{self.model_path}/discriminator.pkl'))
        self.adj_matrix = utils.to_np(torch.load(f'{self.model_path}/adj_matrix.pkl'))
        print('GMM model loaded from {}.'.format(self.model_path))
    
    def get_topic_dist(self, level=0):
        topic_dist = self.G.get_topic_dist(level)
        return topic_dist
    
    def get_topic_word(self, top_k=15, level=-1):

        topic_dist = self.get_topic_dist()
        vals, indices = torch.topk(topic_dist, top_k, dim=1)
        indices = utils.to_np(indices).tolist()
        topic_words = [[self.reader.vocab[idx] for idx in indices[i]]
                       for i in range(topic_dist.shape[0])]
        return topic_words
    
    def eval_level(self):
        trees, relation = utils.build_level(
            self.adj_matrix)
        topic_dist = utils.to_np(self.get_topic_dist())
        topic_word = self.get_topic_word(top_k=10)

        for root in trees.keys():
            level = trees[root]
            for l in level.keys():
                level_topic_dist = topic_dist[level[l]]
                level_topic_word = np.array(topic_word)[level[l]].tolist()

                # 打印主题词
                self.logger.info('\t' * l + f"Level:{l}")
                for k in level[l]:
                    self.logger.info('\t' * l + f'Topic {k}: {topic_word[k]}')
                quality = {}
                quality['TU'] = utils.evaluate_topic_diversity(
                    level_topic_word)
                quality['c_a'] = utils.compute_coherence(
                    self.test_data, level_topic_dist)
                quality['specialization'] = utils.compute_topic_specialization(
                    level_topic_dist, self.test_data)

                self.logger.info('\t' * l + f"Topic quality: {quality}")

        clnpmi = []
        for child in relation.keys():
            father = relation[child]
            child_topic = topic_dist[child]
            father_topic = topic_dist[father]
            clnpmi.append(
                utils.compute_clnpmi(child_topic, father_topic,
                                     self.test_data))
            self.logger.info(
                f"{child}->{father}, clnpmi:{clnpmi[-1]}"
            )
        clnpmi_mean = np.mean(clnpmi)
        self.logger.info(f"Total clnpmi:{clnpmi_mean}")

        
    def evaluate(self):
        topic_dist = utils.to_np(self.get_topic_dist())
        topic_word = self.get_topic_word(top_k=15)
        for k in range(self.topic_num):
            coh_topic = utils.compute_coherence(self.test_data, topic_dist[[k]])
            self.logger.info(
                f'Topic {k} coh[{coh_topic:.3f}]:{topic_word[k][:10]}'
            )

        # self.cur_TU = utils.evaluate_TU(topic_word)
        self.cur_TU = utils.evaluate_TU(topic_dist)
        coherence = {}

        self.logger.info(f"Total TU:{self.cur_TU}, {coherence}")
        return self.cur_TU, coherence

    def check_save(self):
        if self.cur_TU >  self.best_TU:
            if self.cur_coherence > 0.9 * self.best_coherence or self.cur_coherence > 0.285:
                self.best_TU = self.cur_TU
                self.logger.info("New best TU with good coherence found!!")
                self.save_model()
    
    def sample(self, flag=0):
        # get topic_word and print Top word
        topic_dist = utils.to_np(self.get_topic_dist())

        self.cur_coherence = utils.compute_coherence(self.test_data, topic_dist)
        self.logger.info(f"Topic coherence: {self.cur_coherence}")
        if self.cur_coherence > self.best_coherence:
            self.best_coherence = self.cur_coherence
            print("New best coherence found!!")
            self.save_model()

        print(f"Current topic number:{self.topic_num}")
        pass

    def get_noise(self, batch_size):
        noise = self.dirichlet.sample((batch_size,))
        noise = noise.to(device)
        return noise
    
    
    def train(self):
        self.t_begin = time.time()
        # dataloader
        train_dataloader = DataLoader(dataset=MyDataset(self.train_data), batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)


        self.best_coherence = -1
        self.best_TU = 0

        h_all = []

        record_adj = True
        critic_iter = 6 if len(train_dataloader)>5 else len(train_dataloader)
        
        for epoch in tqdm(range(self.epochs)):  

            d_loss_batch = [0]*len(train_dataloader)
            # start = time.perf_counter()
            # print(f'param: {time.perf_counter() - start:.8f}s') 
            e_con_loss = 0.
            """ Discriminator update """
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True
            for p in self.G.parameters():  # recover generator
                p.requires_grad = False
            for p in self.E.parameters():  
                p.requires_grad = False
                
            for idx, data in enumerate(train_dataloader):
                # real_docs = data[:, :-768]
                data = data.cuda()
                if (idx+1) % critic_iter != 0:
                    noise = self.get_noise(self.batch_size)
                    
                    encoder_output, logits, labels = self.E(data)
                    
                    self.D.zero_grad(set_to_none=True)
                    d_loss = torch.mean(self.fake_loss(noise)) - torch.mean(self.real_loss(data, encoder_output))
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    e_con_loss += self.ce(logits, labels)
                    
                    d_loss_batch[idx] = utils.to_np(d_loss)
                    

                else:
                    dis_loss = np.mean(d_loss_batch)
                    # d_loss_all.append(dis_loss)
                    
                    """ Generator and Encoder update """
                    for p in self.D.parameters():
                        p.requires_grad = False  # to avoid computation
                    for p in self.G.parameters():  # recover generator
                        p.requires_grad = True
                    for p in self.E.parameters():  
                        p.requires_grad = True 


                    self.G.zero_grad(set_to_none=True)
                    
                    noise = self.get_noise(self.batch_size)
                    # train generator and adj matrix
                    g_loss = -torch.mean(self.fake_loss(noise))
                    h, f = self.adj_loss()
                    t_loss = g_loss + self.phi * f + 0.5 * self.rho * h * h + self.epsilon * h 
                    t_loss.backward()
                    self.g_optimizer.step()
                    h_all.append(utils.to_np(h))

                    
                    # train encoder
                    self.E.zero_grad(set_to_none=True)
                    encoder_output, logits, labels = self.E(data)
                    e_loss = torch.mean(self.real_loss(data, encoder_output))
                    e_con_loss += self.ce(logits, labels)
                    encoder_loss = e_loss + 100*e_con_loss
                    encoder_loss.backward()
                    self.e_optimizer.step()
                    
                    break
                    
            if epoch > 3000 and (epoch + 1) % 300 == 0:
                # update $\rho$ and $\alpha$
                if h > 0.25 * h_all[-1] and self.rho < self.rho_max:
                    self.rho *= 2
                    self.epsilon += self.rho * h.item()

                # update G Gamma and Temperature
                self.G.gamma = 0.5 + 1 / 2 * np.exp(-0.002 * (epoch - 3000))
                self.G.temperature = 5 * np.exp(-0.002 * (epoch - 3000)) + 1e-5

            
            if epoch % 50 == 0:
                print(
                    f'Epoch: {epoch}/{self.epochs}, d_loss: {dis_loss:.3f}, g_loss: {g_loss:.3f}, ' + 
                    f'e_con_loss: {e_con_loss:.3f}, f: {f:.4f}, h: {h:.4f}'
                )

            if (epoch + 1) % 200 == 0:
                self.sample()

            if (epoch + 1) % 400 == 0:
                self.evaluate()
                self.adj_matrix = utils.to_np(self.G.adjacency_matrix)
                if epoch > 6000:
                    self.check_save()

            if (epoch + 1) % 1000 == 0 and h == 0:
                self.eval_level()
            
            if  (epoch + 1) % 500 == 0 and record_adj and epoch < 10000:
                table_path = os.path.join(args.result_dir, args.dataset + '_' + str(args.topic_num), 'adj_matrix')
                utils.save_adj_matrix(self.adj_matrix, epoch, self.G.temperature , table_path)


        self.t_end = time.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

    # 模型测试
    def test(self):
        self.load_model()
        self.best_coherence = 999
        self.best_TU = 1

        self.evaluate()
        self.sample()
        self.eval_level()
        torch.save(self.G.topic_embed, f'{self.model_path}/topic_emb.pkl')


def main(args=None, logger=None, **kwargs):
    reader = TextReader(args.dataset, args.data_dir)
    if args.emb_type == 'glove':
        emb_path = f"./glove.6B.{args.emb_size}d.txt"
        embedding_mat = utils.build_embedding(emb_path, reader.vocab,
                                              os.path.join(args.result_dir, 'emb', args.emb_type, args.dataset))[:-1]
    else:
        embedding_mat = None
    
    model = GMM(
        args=args,
        reader=reader,
        emb_mat=embedding_mat,
        logger=logger,
        **kwargs
    )
    if args.mode == 'Train':
        model.train()
    elif args.mode == 'Test':
        model.test()
    else:
        print(f'Unknowned mode {args.mode}!')

if __name__ == '__main__':
    logger = utils.log_init(os.path.join(args.result_dir, 'log'), args.dataset + '_' + str(args.topic_num), mode='a')
    message = ''
    index = 0
    for k, v in vars(args).items():
        message = message + str(k) + ": " + str(v) + ' '
        if index % 5 == 0:
            message += '\n'
    logger.info(message)
    utils.seed_init(args)
    main(args=args, logger=logger)

        
    