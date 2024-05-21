import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import softmax, degree


class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        return x

class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2)+torch.mul(1-gate, x1)
        return x


class E2R(nn.Module):
    def __init__(self, e_hidden, r_hidden, max_heads=8, dropout_rate=0.1):
        super(E2R, self).__init__()
        self.a_h_list = nn.ModuleList([nn.Linear(r_hidden, 1, bias=False) for _ in range(max_heads)])
        self.a_t_list = nn.ModuleList([nn.Linear(r_hidden, 1, bias=False) for _ in range(max_heads)])
        self.w_h = nn.Linear(e_hidden, r_hidden, bias=False)
        self.w_t = nn.Linear(e_hidden, r_hidden, bias=False)
        self.head_selector = nn.Parameter(torch.randn(max_heads), requires_grad=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x_e, edge_index, rel, num_heads=None):
        if num_heads is None:
            num_heads = min(int(self.head_selector.softmax(dim=0).sum().item() + 1), len(self.a_h_list))

        edge_index_h, edge_index_t = edge_index
        x_r_h = self.w_h(x_e)
        x_r_t = self.w_t(x_e)

        e1s = [ah(x_r_h).squeeze()[edge_index_h] + at(x_r_t).squeeze()[edge_index_t] for ah, at in
               zip(self.a_h_list[:num_heads], self.a_t_list[:num_heads])]
        alphas1 = [softmax(self.leaky_relu(e1.float()), rel) for e1 in e1s]

        x_r_hs = [spmm(torch.cat([rel.view(1, -1), edge_index_h.view(1, -1)], dim=0), alpha, rel.max() + 1, x_e.size(0),
                       x_r_h) for alpha in alphas1]

        # 对e2进行同样处理
        e2s = [ah(x_r_t).squeeze()[edge_index_h] + at(x_r_h).squeeze()[edge_index_t] for ah, at in
               zip(self.a_h_list[:num_heads], self.a_t_list[:num_heads])]
        alphas2 = [softmax(self.leaky_relu(e2.float()), rel) for e2 in e2s]

        x_r_ts = [spmm(torch.cat([rel.view(1, -1), edge_index_t.view(1, -1)], dim=0), alpha, rel.max() + 1, x_e.size(0),
                       x_r_t) for alpha in alphas2]

        # 将x_r_h和x_r_t的结果相加得到最终x_r，并考虑动态选择的注意力头数量
        x_r = sum(x_r_hs + x_r_ts) / num_heads / 2

        return self.dropout(x_r)


class R2E(nn.Module):
    def __init__(self, e_hidden, r_hidden, num_heads=8, dropout_rate=0.1, residual_connection=False):
        super(R2E, self).__init__()
        self.a_h_list = nn.ModuleList([nn.Linear(e_hidden, 1, bias=False) for _ in range(num_heads)])
        self.a_t_list = nn.ModuleList([nn.Linear(e_hidden, 1, bias=False) for _ in range(num_heads)])
        self.a_r_list = nn.ModuleList([nn.Linear(r_hidden, 1, bias=False) for _ in range(num_heads)])
        self.residual = residual_connection
        if residual_connection:
            self.residual_linear = nn.Linear(e_hidden + r_hidden, e_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x_e, x_r, edge_index, rel):
        edge_index_h, edge_index_t = edge_index
        e_hs = [ah(x_e).squeeze()[edge_index_h] for ah in self.a_h_list]
        e_ts = [at(x_e).squeeze()[edge_index_t] for at in self.a_t_list]
        e_rs = [ar(x_r).squeeze()[rel] for ar in self.a_r_list]

        all_combinations = [(eh + er, et + er) for eh, et, er in zip(e_hs, e_ts, e_rs)]
        alphas_h = [softmax(self.leaky_relu(combination[0]).float(), edge_index_h) for combination in all_combinations]
        alphas_t = [softmax(self.leaky_relu(combination[1]).float(), edge_index_t) for combination in all_combinations]

        x_e_hs = [
            spmm(torch.cat([edge_index_h.view(1, -1), rel.view(1, -1)], dim=0), alpha_h, x_e.size(0), x_r.size(0), x_r)
            for alpha_h in alphas_h]
        x_e_ts = [
            spmm(torch.cat([edge_index_t.view(1, -1), rel.view(1, -1)], dim=0), alpha_t, x_e.size(0), x_r.size(0), x_r)
            for alpha_t in alphas_t]

        x_e_h = sum(x_e_hs) / len(alphas_h)
        x_e_t = sum(x_e_ts) / len(alphas_t)
        x = torch.cat([self.dropout(x_e_h), self.dropout(x_e_t)], dim=1)

        if self.residual:
            x = self.residual_linear(x) + x_e

        return x


class GAT(nn.Module):
    def __init__(self, hidden, num_heads=8, dropout_rate=0.1):
        super(GAT, self).__init__()
        self.a_i_list = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(num_heads)])
        self.a_j_list = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        e_ij = [ai(x).squeeze()[edge_index_i] + aj(x).squeeze()[edge_index_j] for ai, aj in zip(self.a_i_list, self.a_j_list)]
        alpha = [softmax(F.leaky_relu(eij.float()), edge_index_i) for eij in e_ij]
        x = self.dropout(sum(spmm(edge_index[[1, 0]], alpha_head, x.size(0), x.size(0), x) for alpha_head in alpha) / len(alpha))
        return x


class InteractiveGCN(nn.Module):
    def __init__(self, e_hidden=300, r_hidden=100,  gat_dropout=0.1,
                 max_gat_heads=8):
        super(InteractiveGCN, self).__init__()

        self.gcn1 = GCN(e_hidden)
        self.highway1 = Highway(e_hidden)
        self.gcn2 = GCN(e_hidden)
        self.highway2 = Highway(e_hidden)

        self.e2r = E2R(e_hidden, r_hidden, max_heads=max_gat_heads, dropout_rate=gat_dropout)
        self.r2e = R2E(e_hidden, r_hidden, num_heads=max_gat_heads, dropout_rate=gat_dropout)
        self.gat = GAT(e_hidden + 2 * r_hidden, num_heads=max_gat_heads, dropout_rate=gat_dropout)

    def forward(self, x_e, edge_index, rel, edge_index_all, rel_all):
        x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
        x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))
        x_r = self.e2r(x_e, edge_index, rel)
        x_e = torch.cat([x_e, self.r2e(x_e, x_r, edge_index, rel)], dim=1)
        x_e = torch.cat([x_e, self.gat(x_e, edge_index_all)], dim=1)
        return x_e