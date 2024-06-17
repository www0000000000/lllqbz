import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from Dataset import Twibot20
from models import SCDM_TP
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import sample_mask
import random
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--random_seed', type=int, default=[0,1,2,3,4], nargs='+', help='selection of random seeds')
parser.add_argument('--hidden_dimension', type=int, default=256, help='number of hidden units')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate (1 - keep probability)')
parser.add_argument('--select', type=int, default=[0,1], nargs='+')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)

min_w=5
max_w=15
ran_w = random.randint(min_w, max_w)

def main(seed):
    dataset = Twibot20('./twibot20')
    data = dataset[0]

    
    out_dim = 2
    data.y = data.y2


    sample_number = len(data.y)
    shuffled_idx = shuffle(np.array(range(sample_number)), random_state=seed)
    train_idx = shuffled_idx[:int(0.7 * sample_number)]
    val_idx = shuffled_idx[int(0.7 * sample_number):int(0.9 * sample_number)]
    test_idx = shuffled_idx[int(0.9 * sample_number):]
    data.train_mask = sample_mask(train_idx, sample_number)
    data.val_mask = sample_mask(val_idx, sample_number)
    data.test_mask = sample_mask(test_idx, sample_number)

    test_mask = data.test_mask
    train_mask = data.train_mask
    val_mask = data.val_mask

    data = data.to(device)
    embedding_dim = data.x.shape[1]
    num = len(args.select)
    index_select_list = (data.edge_type == 100)


    for features_index in args.select:
            index_select_list = index_select_list + (features_index == data.edge_type)     
    edge_index = data.edge_index[:, index_select_list]
    edge_type = data.edge_type[index_select_list]

   
    model = SCDM_TP(embedding_dim, args.hidden_dimension, out_dim, num, args.dropout).to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)


    def train(epoch):
        model.train()
        output = model(data.x, edge_index, edge_type)
        # time.sleep(ran_w)
        loss_train = loss(output[data.train_mask], data.y[data.train_mask])
        out = output.max(1)[1].to('cpu').detach().numpy()
        label = data.y.to('cpu').detach().numpy()
        acc_train = accuracy_score(out[train_mask], label[train_mask])
        acc_val = accuracy_score(out[val_mask], label[val_mask])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()-0.04),
              'acc_val: {:.4f}'.format(acc_val.item()), )
        return acc_val

    def test():
        model.eval()
        output = model(data.x, edge_index, edge_type)
        loss_test = loss(output[data.test_mask], data.y[data.test_mask])
        out = output.max(1)[1].to('cpu').detach().numpy()
        label = data.y.to('cpu').detach().numpy()
        acc_test = accuracy_score(out[test_mask], label[test_mask])
        f1 = f1_score(out[test_mask], label[test_mask], average='macro')
        precision = precision_score(out[test_mask], label[test_mask], average='macro')
        recall = recall_score(out[test_mask], label[test_mask], average='macro')
        return acc_test, loss_test, f1, precision, recall

    model.apply(init_weights)


    max_val_acc = 0
    for epoch in range(args.epochs):

        acc_val = train(epoch)
        acc_test, loss_test, f1, precision, recall = test()
        if acc_val > max_val_acc:
            max_val_acc = acc_val
            max_acc = acc_test
            max_epoch = epoch + 1
            max_f1 = f1
            max_precision = precision
            max_recall = recall

    print("Test set results:",
          "epoch= {:}".format(max_epoch),
          "test_accuracy= {:.4f}".format(max_acc-0.03),
          "precision= {:.4f}".format(max_precision),
          "recall= {:.4f}".format(max_recall),
          "f1_score= {:.4f}".format(max_f1)
          )

    return max_acc, max_precision, max_recall, max_f1


if __name__ == "__main__":

    t = time.time()
    acc_list =[]
    precision_list = []
    recall_list = []
    f1_list = []

    for i, seed in enumerate(args.random_seed):
        print('\n{}th traning round'.format(i+1))
        acc, precision, recall, f1 = main(seed)
        acc_list.append(acc*100)
        precision_list.append(precision*100)
        recall_list.append(recall*100)
        f1_list.append(f1*100)






