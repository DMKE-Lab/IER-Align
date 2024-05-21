import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
import numpy as np
from multiprocessing import Pool


def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all


def load_alignment_pair(file_name):
    alignment_pair = []
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair


def load_data(path):
    entity1, rel1, time1, quadruples1 = load_quadruples(path + '/triples_1')
    entity2, rel2, time2, quadruples2 = load_quadruples(path + '/triples_2')
    dev_pair = load_alignment_pair(path + '/ref_ent_ids')

    time_point, time_interval = get_matrix(quadruples1 + quadruples2, entity1.union(entity2), rel1.union(rel2), time1.union(time2))
    return np.array(dev_pair),time_point, time_interval


thread_num=18


def intersection_length(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    return max(0, end - start + 1)


def sim_matrix(t1, t2):
    size_t1 = len(t1)
    size_t2 = len(t2)
    matrix = np.zeros((size_t1, size_t2))

    for i in range(size_t1):
        time_intervals_i = [key for key, value in t1[i].items() if isinstance(key, list) for _ in range(value)]
        time_points_i = [key for key, value in t1[i].items() if not isinstance(key, list) for _ in range(value)]

        for j in range(size_t2):
            time_intervals_j = [key for key, value in t2[j].items() if isinstance(key, list) for _ in range(value)]
            time_points_j = [key for key, value in t2[j].items() if not isinstance(key, list) for _ in range(value)]

            point_overlap = len(set(time_points_i).intersection(set(time_points_j)))
            interval_overlap = 0
            for interval1 in time_intervals_i:
                for interval2 in time_intervals_j:
                    interval_overlap += intersection_length(interval1, interval2)

            total_length_interval = sum([sub_interval[1] - sub_interval[0] + 1 for sub_interval in time_intervals_i]) + sum(
                [sub_interval[1] - sub_interval[0] + 1 for sub_interval in time_intervals_j])
            total_length_point = len(time_points_i) + len(time_points_j)

            if total_length_point==0 :
                if total_length_interval == 0:
                    sim = 0
                else:
                    sim_intervals = interval_overlap / total_length_interval
                    sim = sim_intervals
            elif total_length_interval == 0:
                sim_point = point_overlap / total_length_point
                sim = sim_point
            else:
                sim_point = point_overlap / total_length_point
                sim_intervals = interval_overlap / total_length_interval
                sim = (sim_point + sim_intervals) / 2

            matrix[i, j] = sim
    return matrix


def thread_sim_matrix(t1,t2):
    pool = Pool(processes=thread_num)
    reses = list()
    tasks_t1 = div_array(t1,thread_num)
    for task_t1 in tasks_t1:
        reses.append(pool.apply_async(sim_matrix,args=(task_t1,t2)))
    pool.close()
    pool.join()
    matrix = None
    for res in reses:
        val = res.get()
        if matrix is None:
            matrix = np.array(val)
        else:
            matrix = np.concatenate((matrix,val),axis=0)
    return matrix


def list2dict(time_list):
    dic = {}
    for i in time_list:
        if isinstance(i, list):
            key = tuple(i)
        else:
            key = i
        dic[key] = time_list.count(i)
    return dic


def get_matrix(triples, entity, rel, time):
    time_point = {}
    time_interval = {}

    for i in range(max(entity) + 1):
        time_point[i] = []
        time_interval[i] = []

    for h, r, t, ts, te in triples:
        if ts == te:
            time_point[h].append(ts);
            time_point[t].append(ts)
        elif ts == 0:
            time_point[h].append(te);
            time_point[t].append(te)
        elif te == 0:
            time_point[h].append(ts);
            time_point[t].append(ts)
        else:
            time_interval[h].append([ts, te]);
            time_interval[t].append([ts, te])

    return time_point, time_interval


def load_quadruples(file_name):
    quadruples = []
    entity = set()
    rel = set([0])
    time = set()
    for line in open(file_name, 'r'):
        items = line.split()
        if len(items) == 4:
            head, r, tail, t = [int(item) for item in items]
            entity.add(head);
            entity.add(tail);
            rel.add(r);
            time.add(t)
            quadruples.append((head, r, tail, t, t))
        else:
            head, r, tail, tb, te = [int(item) for item in items]
            entity.add(head);
            entity.add(tail);
            rel.add(r);
            time.add(tb);
            time.add(te)
            quadruples.append((head, r, tail, tb, te))
    return entity, rel, time, quadruples


def get_train_batch(x1, x2, train_set, k=5):
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2], dim=0)
    return train_batch


def get_hits(x1, x2, pair, dist='L1', Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    print(S.size())
    print('Left:\t',end='')
    for k in Hn_nums:
        pred_topk= S.topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)
    print('Right:\t',end='')
    for k in Hn_nums:
        pred_topk= S.t().topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.t().sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)

    
def get_hits_stable(x1, x2, pair):
    pair_num = pair.size(0)
    S = -torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1).cpu()
    #index = S.flatten().argsort(descending=True)
    index = (S.softmax(1)+S.softmax(0)).flatten().argsort(descending=True)
    index_e1 = index//pair_num
    index_e2 = index%pair_num
    aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
    aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
    true_aligned = 0
    for _ in range(pair_num*100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    print('Both:\tHits@Stable: %.2f%%    ' % (true_aligned/pair_num*100))
