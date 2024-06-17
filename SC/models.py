import torch
import numpy as np
from torch import nn
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
import pandas as pd
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.distributed as dist
import time


class LSTM(nn.Module):
    def __init__(self,input_size=16,hidden_size=256,output_size=3,num_layer=1,dropout=0.3,bidirectional=True):
        super(LSTM,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer,dropout,bidirectional)
        self.layer2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x

class Attention(nn.Module):

    def __init__(self, vector_size):
        super(Attention, self).__init__()
        self.vector_size = vector_size

        self.fc = nn.Linear(vector_size, vector_size)
        self.weightparam = nn.Parameter(torch.randn(vector_size, 1))

    def forward(self, vectors):
        weight = torch.tanh(self.fc(vectors)).matmul(self.weightparam)
        weight = F.softmax(weight, dim=0)
        rep = vectors.mul(weight)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        rep = rep.sum(dim=0)
        return rep


class SemanticWord(nn.Module):

    def __init__(self, embedding_dim, rep_size, batch_size, num_layer, embed_layer, p):
        super(SemanticWord, self).__init__()
        self.hidden_dim = int(rep_size / 2)
        self.embedding_dim = embedding_dim
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.num_layer = num_layer

        self.word_embeddings = embed_layer
        self.word_embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, dropout=p, num_layers=num_layer)
        

    def init_hidden(self, batch_size):
        
        temp = (torch.cuda.FloatTensor(2 * self.num_layer, batch_size, self.hidden_dim).fill_(0),
                torch.cuda.FloatTensor(2 * self.num_layer, batch_size, self.hidden_dim).fill_(0))
        return temp
        
    def forward(self, text):
        sim_batch_size = 8
        batch_size = len(text[0])
        if batch_size <= sim_batch_size:
            self.hidden = self.init_hidden(batch_size)
    
            tmp = list(set(range(len(text))) - tmp)
            if not len(tmp):
                tmp = [0]
            text = text[tmp]
            result = self.word_embeddings(text)
            result = result.clone().view(len(text), batch_size, -1).cuda(non_blocking=True)
            result, _ = self.lstm(result, self.hidden)
            del self.hidden
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return result
        else:
            now = 0
            tmp = []
           
            while True:
                now_text = text[:, now:min(now + sim_batch_size, batch_size)]
                now_batch_size = len(now_text[0])
                
                self.hidden = self.init_hidden(now_batch_size)
                result = self.word_embeddings(now_text)
                result = result.clone().view(len(now_text), now_batch_size, -1).cuda(non_blocking=True)
                result, _ = self.lstm(result, self.hidden)
               
                del now_text
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                tmp.append(result)
                if now + sim_batch_size >= batch_size:
                    break
                now += sim_batch_size
            tmp = torch.cat(tmp, dim=1)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # print('before attention: ', tmp.size())
            return tmp



class SemanticTweet(nn.Module):

    def __init__(self, input_dim, rep_size, num_layer, p):
        super(SemanticTweet, self).__init__()
        self.hidden_dim = int(rep_size / 2)
        self.input_dim = input_dim
        self.rep_size = rep_size
        self.batch_size = 1
        self.num_layer = num_layer

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, bidirectional=True, dropout=p, num_layers=num_layer)
        self.hidden = self.init_hidden()
        

    def init_hidden(self):
        temp = (torch.cuda.FloatTensor(2 * self.num_layer, self.batch_size, self.hidden_dim).fill_(0),
                torch.cuda.FloatTensor(2 * self.num_layer, self.batch_size, self.hidden_dim).fill_(0))
        return temp
       
    def forward(self, vectors):
        self.hidden = self.init_hidden()
      
        result, _ = self.lstm(vectors, self.hidden)
        result = result.squeeze(1)
       
        return result



class SemanticVector(nn.Module):

    def __init__(self, embedding_dim, rep_size, num_layer, dropout, embed_layer):
        super(SemanticVector, self).__init__()
        self.embedding_dim = embedding_dim
        self.rep_size = rep_size
        self.num_layer = num_layer
        self.dropout = dropout
        # self.embed_layer = embed_layer
        # self.embed_layer.weight.requires_grad = False

        self.WordLevelModel = SemanticWord(embedding_dim=self.embedding_dim, rep_size=int(self.rep_size / 2),
                                           batch_size=1,
                                           num_layer=self.num_layer, embed_layer=embed_layer, p=self.dropout)
        self.TweetLowModel = SemanticWord(embedding_dim=self.embedding_dim, rep_size=int(self.rep_size / 2),
                                          batch_size=1,
                                          num_layer=self.num_layer, embed_layer=embed_layer, p=self.dropout)
        self.TweetHighModel = SemanticTweet(input_dim=int(self.rep_size / 2), rep_size=int(self.rep_size / 2),
                                            num_layer=self.num_layer, p=dropout)
        self.WordAttention = Attention(vector_size=int(self.rep_size / 2))
        self.TweetLowAttention = Attention(vector_size=int(self.rep_size / 2))
        self.TweetHighAttention = Attention(vector_size=int(self.rep_size / 2))

    def forward(self, user):
        text_word = user['word']
        # text_word = text_word.unsqueeze(1)
        WordLevelRep = self.WordAttention(self.WordLevelModel(text_word.unsqueeze(1)).squeeze(1))
        del text_word
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        text_tweet = user['tweet']  # one tweet each row
        TweetRep = self.TweetLowAttention(self.TweetLowModel(text_tweet.transpose(0, 1)))
        del text_tweet
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        TweetLevelRep = self.TweetHighAttention(self.TweetHighModel(TweetRep.unsqueeze(1)))
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return torch.cat((WordLevelRep, TweetLevelRep))


class Properties(nn.Module):

    def __init__(self, input_size, rep_size, dropout):
        super(Properties, self).__init__()
        self.input_size = input_size
        self.rep_size = rep_size
        self.dropout = dropout
       
        self.fc = nn.Linear(self.input_size, self.rep_size)
        self.act = nn.ReLU()

    def forward(self, vectors):
        vectors = self.act(self.fc(vectors))
        return vectors

reps = []
matrix0 = []
matrix1 = []

def max_pool(matrix, pool_size):

    rows, cols = matrix.shape

    out_rows = (rows - pool_size) // 2 + 1
    out_cols = (cols - pool_size) // 2 + 1
    
    pooled = np.empty((out_rows, out_cols))

    for i in range(out_rows):
        for j in range(out_cols):
            region = matrix[i:i+pool_size, j:j+pool_size]
            pooled[i, j] = np.max(region)
    
    return pooled

class SenCon(nn.Module):
    def __init__(self, embed_dim,embed_layer,num_layer,dropout):
        super(SenCon, self).__init__()
        self.embed_dim=embed_dim
        self.embed_layer = embed_layer
        self.num_layer = num_layer
        self.dropout = dropout

    def getSC(self,queries,keys,hidden_size):
        attention_vec = np.matmul(queries, keys.transpose((0, 2, 1))) / np.sqrt(hidden_size)
        attention_vec = np.exp(attention_vec) / np.sum(np.exp(attention_vec), axis=1, keepdims=True)
        pool_weights=max_pool(attention_vec, size=(9, 9))
        semantic_matrix=pool_weights[4:-4, 4:-4]
        sencon_vec=semantic_matrix.flatten()

        return sencon_vec

class CoAttention(nn.Module):

    def __init__(self, vec_size):
        super(CoAttention, self).__init__()
        self.vec_size = vec_size
        self.Wsp = nn.Parameter(torch.randn(self.vec_size, self.vec_size))
        self.Wpn = nn.Parameter(torch.randn(self.vec_size, 2 * self.vec_size))
        self.Wns = nn.Parameter(torch.randn(2 * self.vec_size, self.vec_size))
        self.Ws = nn.Parameter(torch.randn(self.vec_size, self.vec_size))
        self.Wp = nn.Parameter(torch.randn(self.vec_size, self.vec_size))
        self.Wn = nn.Parameter(torch.randn(self.vec_size, 2 * self.vec_size))
        self.Wh = nn.Parameter(torch.randn(3 * self.vec_size, self.vec_size))
        self.fc = nn.Linear(2 * vec_size, 2 * vec_size)
        self.act = nn.ReLU()

    def forward(self, user):
        
        Vs = torch.transpose(user['semantic'], 1, 2)
        Vp = torch.transpose(user['property'], 1, 2)
        Vn = torch.transpose(self.act(self.fc(user['consistency'])), 1, 2)
        
        Fsp = torch.tanh(torch.transpose(Vs, 1, 2).matmul(self.Wsp).matmul(Vp))
        Fpn = torch.tanh(torch.transpose(Vp, 1, 2).matmul(self.Wpn).matmul(Vn))
        Fns = torch.tanh(torch.transpose(Vn, 1, 2).matmul(self.Wns).matmul(Vs))

        Hs = torch.tanh(self.Ws.matmul(Vs) + self.Wp.matmul(Vp) * Fsp + self.Wn.matmul(Vn) * Fns)
        Hp = torch.tanh(self.Wp.matmul(Vp) + self.Ws.matmul(Vs) * Fsp + self.Wn.matmul(Vn) * Fpn)
        Hn = torch.tanh(self.Wn.matmul(Vn) + self.Ws.matmul(Vs) * Fns + self.Wp.matmul(Vp) * Fpn)

        H = torch.cat((torch.cat((Hs, Hp), dim = 1), Hn), dim = 1)

        result = torch.tanh(torch.transpose(H, 1, 2).matmul(self.Wh))
        del Vs,Vp,Vn,Fsp,Fpn,Fns,Hs,Hp,Hn,H
        return result

class Classification(nn.Module):

    def __init__(self, vec_size, label_size, dropout):
        super(Classification, self).__init__()
        self.vec_size = vec_size
        self.label_size = label_size

        # self.fc1 = nn.Linear(self.vec_size, self.vec_size)
        self.fc2 = nn.Linear(self.vec_size, self.label_size)
        # self.fc1.bias.data.fill_(0)
        # self.fc2.bias.data.fill_(0)
        # self.act1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(p=dropout)

    def forward(self, vector):
        # result = self.dropout1(self.act1(self.fc1(vector)))
        # print(result.size())
        result = F.log_softmax(self.fc2(vector), dim=1)
        return result



class ModelBatch(nn.Module):

    def __init__(self, EMBEDDING_DIM, REP_SIZE, NUM_LAYER, DROPOUT, EMBED_LAYER, PROPERTY_SIZE,LABEL_SIZE ):
        super(ModelBatch, self).__init__()

        self.SemanticModel = SemanticVector(embedding_dim=EMBEDDING_DIM, rep_size=REP_SIZE, num_layer=NUM_LAYER,
                                            dropout=DROPOUT, embed_layer=EMBED_LAYER)
        self.PropertyModel = Properties(input_size=PROPERTY_SIZE, dropout=DROPOUT, rep_size=REP_SIZE)
        self.SenConModel =SenCon(embedding_dim=EMBEDDING_DIM,num_layer=NUM_LAYER,dropout=DROPOUT,embed_layer=EMBED_LAYER)
        self.CoAttentionModel = CoAttention(vec_size=REP_SIZE)
        self.ClassficationModel=Classification(vec_size=REP_SIZE, label_size=LABEL_SIZE, dropout=DROPOUT)

    def forward(self, user_batch):
       
        semantic_reps = []
        for i in range(len(user_batch['word'])):
            semantic_reps.append(self.SemanticModel({'word': user_batch['word'][i], 'tweet': user_batch['tweet'][i]}))
        

        property_reps = self.PropertyModel(user_batch['property'])

        vector_reps = self.CoAttentionModel({'semantic': torch.stack(semantic_reps).unsqueeze(1),
                                             'property': property_reps,
                                             'consistency': user_batch['consistency']}).squeeze(1)

       
        result = self.ClassficationModel(vector_reps)
     
        return result, semantic_reps, property_reps, vector_reps



class TweetDataset(Dataset):
    def __init__(self, file):
        self.file = file.replace('.txt', '')
        self.IDList = []
        with open(file, 'r', encoding='utf-8') as f:
            for ID in f:
                self.IDList.append(ID.split()[0])
        zeros = torch.zeros([1,600], dtype = torch.float)
        # for ID in self.IDList:
        #     path = 'neighbor_tensor/' + ID + '.pt'
        #     torch.save(zeros, path)

    def __len__(self):
        return len(self.IDList)

    # def __getitem__(self, index):
    #     ID = self.IDList[index]
    #     path = 'tensor/' + ID + '.pt'
    #     user = torch.load(path)
    #     path = 'neighbor_tensor/' + ID + '.pt'
    #     user['neighbor'] = torch.load(path)
    #     return user

    def check(self, ID):
        return ID in self.IDList

    # def update(self, ID, val):
    #     path = 'neighbor_tensor/' + ID + self.file + '.pt'
    #     torch.save(val, path)


# class SCDM_TP(nn.Module):
#     def __init__(self, EMBEDDING_DIM, REP_SIZE, LABEL_SIZE, PROPERTY_SIZE, DROPOUT):
#         super(SCDM_TP, self).__init__()
          
#         self.SCDM_TP = ModelBatch(EMBEDDING_DIM = EMBEDDING_DIM, REP_SIZE = REP_SIZE, NUM_LAYER = 2, DROPOUT = DROPOUT,
#                                 EMBED_LAYER = 2, PROPERTY_SIZE = PROPERTY_SIZE, LABEL_SIZE = LABEL_SIZE)

#         self.fc = nn.Linear(REP_SIZE, 2)

#     def forward(self, user_batch):
#         _, _, _, reps = self.SCDM_TP(user_batch)
#         result = F.log_softmax(self.fc(reps), dim=1)
#         return result


# padding for each batch in DataLoader
def pad_collate(batch):
    final = {}
    tmp = []
    for user in batch:
        tmp.append(user['word'])
    tmp = torch.nn.utils.rnn.pad_sequence(tmp, batch_first=True, padding_value=tdict['b_b'])
    final['word'] = tmp
    tmp = []
    for user in batch:
        tmp.append(user['id'])
    final['id'] = torch.stack(tmp)
    tmp = []
    for user in batch:
        tmp.append(user['target'])

    final['target'] = torch.stack(tmp)


    tmp = []
    for user in batch:
        tmp.append(user['property'])
    final['property'] = torch.stack(tmp)
    mxH = 0
    mxL = 0
    for user in batch:
        mxH = max(mxH, user['tweet'].size()[0])
        mxL = max(mxL, user['tweet'].size()[1])
    empty = [dict] * mxL
    tmp = []
    for user in batch:
        T = []
        tweet = user['tweet'].numpy().tolist()
        for i in tweet:
            T.append(i)
        for i in range(mxH - len(tweet)):
            T.append(empty)
        tmp.append(T)
    final['tweet'] = torch.tensor(tmp)
    return final


class SCDM_TP(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, num=2, dropout=0.3):
        super(SCDM_TP, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.lstm1 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=num)
        self.lstm2 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=num)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)


    def forward(self, feature, index, consisenty):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.lstm1(x, index, consisenty)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lstm1(x, index, consisenty)
        x = self.linear_output2(x) 


        return x




    


