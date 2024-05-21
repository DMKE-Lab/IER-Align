import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import InteractiveGCN
from data import DBP15K
from loss import L1_Loss
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        default='1',
        type=str,
        help='choose gpu device')
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", default="data/DBP15K")
    parser.add_argument("--lang", default="zh_en")
    parser.add_argument("--rate", type=float, default=0.3)

    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--neg_epoch", type=int, default=10)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--stable_test", action="store_true", default=True)
    args = parser.parse_args()
    return args


def init_data(args, device):
    data = DBP15K(args.data, args.lang, rate=args.rate)[0]
    data.x1 = F.normalize(data.x1, dim=1, p=2).to(device).requires_grad_()
    data.x2 = F.normalize(data.x2, dim=1, p=2).to(device).requires_grad_()
    data.edge_index_all1, data.rel_all1 = add_inverse_rels(data.edge_index1, data.rel1)
    data.edge_index_all2, data.rel_all2 = add_inverse_rels(data.edge_index2, data.rel2)
    return data


def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
        x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    return x1, x2
    

def train(model, criterion, optimizer, data, train_batch):
    model.train()
    # 对两个图分别进行前向传播
    x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
    x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    # 计算损失
    loss = criterion(x1, x2, data.train_set, train_batch)
    # 清零梯度
    optimizer.zero_grad()
    # 反向传播并计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()
    return loss


def main(args):
    device = torch.device('cpu:' + args.gpu)
    data = init_data(args, device).to(device)
    dev_pair,time_point, time_interval = load_data(args.data +'/'+args.lang)

    rest_set_1 = [e1 for e1, e2 in dev_pair]
    rest_set_2 = [e2 for e1, e2 in dev_pair]
    np.random.shuffle(rest_set_1)
    np.random.shuffle(rest_set_2)
    t1 = []
    for e1 in rest_set_1:
        # 处理时间点
        tp1 = time_point.get(e1, [])
        t1.append(list2dict(tp1))
        # 处理时间间隔
        ti1 = time_interval.get(e1, [])
        t1[-1].update(list2dict(ti1))

    t2 = []
    for e2 in rest_set_2:
        # 处理时间点
        tp2 = time_point.get(e2, [])
        t2.append(list2dict(tp2))
        # 处理时间间隔
        ti2 = time_interval.get(e2, [])
        t2[-1].update(list2dict(ti2))

    m = thread_sim_matrix(t1, t2)

    model = InteractiveGCN(data.x1.size(1), args.r_hidden).to(device)

    # 将模型参数与可训练的数据张量一同添加到优化器中

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    criterion = L1_Loss(args.gamma)

    for epoch in range(args.epoch):
        if epoch % args.neg_epoch == 0:
            x1, x2 = get_emb(model, data)
            train_batch = get_train_batch(x1, x2, data.train_set, args.k)

        loss = train(model, criterion, optimizer, data, train_batch)
        print('Epoch:', epoch + 1, '/', args.epoch, '\tLoss: %.3f' % loss, '\r', end='')

        if (epoch + 1) % args.test_epoch == 0:
            print()
            test(model, data, args.stable_test, m, args.alpha)

    
if __name__ == '__main__':
    args = parse_args()
    main(args)
