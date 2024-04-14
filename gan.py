import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import geoopt.manifolds.stereographic.math as pmath_geo


dist_norm = Normal(0., 1.)


class Encoder(torch.nn.Module):
    def __init__(self, args=None, hidden_num=400, is_training=True, device=None, **kwargs):
        super(Encoder, self).__init__()
        self.K = 1024
        self.m = 0.999
        self.T = 0.07
        self.dropout = args.dropout
        self.is_training = is_training
        self.device = device
        self.sim = nn.CosineSimilarity(dim=-1)
        self.main_module = nn.Sequential(nn.Linear(args.vocab_size, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, hidden_num),
                                         nn.BatchNorm1d(hidden_num),
                                         nn.LeakyReLU(),
                                         nn.Linear(hidden_num, args.topic_num))
        self.main_module_con = nn.Sequential(nn.Linear(args.vocab_size, 1024),
                                             nn.BatchNorm1d(1024),
                                             nn.LeakyReLU(),
                                             nn.Linear(1024, hidden_num),
                                             nn.BatchNorm1d(hidden_num),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_num, args.topic_num))
        self.softmax = nn.Softmax(dim=1)
        
        for param_q, param_k in zip(self.main_module.parameters(), self.main_module_con.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        self.register_buffer("queue", torch.randn(args.topic_num, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def set_use_cos(self, is_use):
        self.use_cos = is_use
    
    def is_test(self):
        self.is_training = False
        
        
    def simcse_encoder(self, query, key):
        query = self.main_module(query)
        q = F.normalize(query, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            key = self.main_module_con(key)
            k = F.normalize(key, dim=1)
            
        l_pos = self.sim(q.unsqueeze(1), k.unsqueeze(0)) / 0.05
        l_neg = self.sim(q.unsqueeze(1), self.queue.clone().detach().T.unsqueeze(0)) / 0.05

        logits = torch.cat((l_pos, l_neg), dim=1)
        labels = torch.arange(logits.shape[0], dtype=torch.long).to(self.device)
        
        # dequeue and enqueue
        if self.is_training:
            self._dequeue_and_enqueue(k)
        
        return logits, labels
    
    def forward(self, x):
        query, key = self.data_aug(x)
        logits, labels = self.simcse_encoder(query, key)    
        return self.softmax(self.main_module(x)), logits, labels
    
    @torch.no_grad()
    def data_aug(self, x):
        query = F.dropout(self.set_smaller_values_to_zero(x), p=self.dropout)
        key = F.dropout(self.set_smaller_values_to_zero(x), p=self.dropout)
        return query, key
        
    @torch.no_grad()
    def set_smaller_values_to_zero(self, x):
        row_min_values, _ = torch.min(x, dim=1, keepdim=True)
        row_max_values, _ = torch.max(x, dim=1, keepdim=True)

        probs = (x - row_min_values) / (row_max_values - row_min_values + 1e-9)
        probs = probs.to(self.device)
        mask = torch.rand_like(probs) > probs
        mask_x = x * mask

        return mask_x
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.main_module.parameters(), self.main_module_con.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0 # for simple
        # replace the keys at ptr(dequeue and enqueue)
        self.queue[:, ptr:ptr+batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K # move pointer
        self.queue_ptr[0] = ptr


class Generator(torch.nn.Module):
    def __init__(self,
                 args=None,
                 emb_mat=None,
                 device=None,
                 learnable_curvature=False,
                 **kwargs):
        super(Generator, self).__init__()

        xavier_init = torch.distributions.Uniform(-0.05,0.05)
        self.args = args
        # define embedding
        if emb_mat == None:
            self.word_embed = nn.Parameter(xavier_init.sample((args.vocab_size, args.emb_size)))
        else:
            print("Using pre-train word embedding")
            self.word_embed = nn.Parameter(emb_mat).to(device)
        
        self.topic_embed = nn.Parameter(xavier_init.sample((args.topic_num, args.emb_size))).to(device)
        self.Q_vec_1 = nn.Parameter(xavier_init.sample((int(args.emb_size / 2), int(args.emb_size / 2))))
        self.K_vec_1 = nn.Parameter(xavier_init.sample((int(args.emb_size / 2), int(args.emb_size / 2)))) 
        self.Q_vec_2 = nn.Parameter(xavier_init.sample((int(args.emb_size / 2), int(args.emb_size / 2))))
        self.K_vec_2 = nn.Parameter(xavier_init.sample((int(args.emb_size / 2), int(args.emb_size / 2))))   

        if learnable_curvature:
            # print("Init")
            self.c = torch.nn.Parameter(torch.tensor([1.0]).to(device))
        else:
            self.c = torch.FloatTensor([1.0]).to(device)
        self.ha_1 = HypHawkes(dimensions=int(args.emb_size / 2), bs=args.topic_num, device=device, c=self.c)
        self.ha_2 = HypHawkes(dimensions=int(args.emb_size / 2), bs=args.topic_num, device=device, c=self.c)
        self.linear_in = torch.nn.Linear(int(args.emb_size / 2), int(args.emb_size / 2), bias=False)
        self.gamma = 1
        self.temperature = 5   

        
        

    def get_topic_dist(self, level=0):
        try:
            return self.beta
        except:
            self.beta = torch.softmax(self.topic_embed @ self.word_embed.T,
                                      dim=1)
            return self.beta

    
    def update_adjacency_matrix(self):
        q_1, k_1 = self.topic_embed[:, :150] @ self.Q_vec_1, self.topic_embed[:, :150] @ self.K_vec_1
        q_2, k_2 = self.topic_embed[:, 150:] @ self.Q_vec_2, self.topic_embed[:, 150:] @ self.K_vec_2
        self.adj_1 = self.ha_1(q_1, k_1)
        self.adj_2 = self.ha_2(q_2, k_2)
        self.adjacency_matrix = F.softmax((self.adj_1 + self.adj_2) / self.temperature, dim=1)
        

    def get_adjacency_matrix(self):
        theta_1 = self.pi @ self.beta
        theta_2 = self.pi @ self.adjacency_matrix @ self.beta
        return self.adjacency_matrix, theta_1 - theta_2

    def forward(self, x):
        self.update_adjacency_matrix()

        self.beta = torch.softmax(self.topic_embed @ self.word_embed.T, dim=1)
        self.pi = x * self.gamma
        self.theta = self.pi @ self.beta

        return self.theta
    

class Discriminator(torch.nn.Module):
    def __init__(self, args=None, hidden_num=200, **kwargs):
        super(Discriminator, self).__init__()

        self.main_module = nn.Sequential(
            nn.BatchNorm1d(args.vocab_size + args.topic_num),
            nn.utils.spectral_norm(nn.Linear(args.vocab_size + args.topic_num,
                                             hidden_num)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_num, 1)),
        )

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        return self.main_module(x)
    

# HYPHEN: Hyperbolic Hawkes Attention For Text Streams
class HypHawkes(torch.nn.Module):
    def __init__(self, dimensions, bs, device=None, attention_type="general", c=1.0):
        super(HypHawkes, self).__init__()

        if attention_type not in ["dot", "general"]:
            raise ValueError("Invalid attention type selected.")

        self.attention_type = attention_type
        self.c = torch.nn.Parameter(torch.tensor([c])).to(device)
        if self.attention_type == "general":
            self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()
        self.ae = torch.nn.Parameter(torch.FloatTensor(bs, 1, 1))
        self.ab = torch.nn.Parameter(torch.FloatTensor(bs, 1, 1))

    def forward(self, query, context):
        n, d = query.shape
        query = pmath_geo.expmap0(query, k=self.c)
        query = pmath_geo.project(query, k=self.c)
        
        context = pmath_geo.expmap0(context, k=self.c)
        context = pmath_geo.project(context, k=self.c)
        
        query = self.linear_in(query)

        attention_scores = query @ context.T

        # Compute weights across every context sequence
        attention_weights = self.softmax(attention_scores / d)
        # converting hyp
        attention_weights = pmath_geo.expmap0(attention_weights, k=self.c)
        attention_weights = pmath_geo.project(attention_weights, k=self.c)

        return attention_weights