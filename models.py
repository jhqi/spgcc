import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution

class SPGCC(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_sp, cluster_num, dropout=0.1, threshold=0.8, temperature=0.5, alpha=0.1, device=torch.device('cuda:0')):
        super(SPGCC, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3_1 = GraphConvolution(nhid, nout)
        self.gc3_2 = GraphConvolution(nhid, nout)
        self.n_sp = n_sp
        self.cluster_num = cluster_num
        self.dropout = dropout
        self.threshold = threshold
        self.temperature = temperature
        self.alpha = alpha
        self.device = device

    @staticmethod
    def neighbour_alignment_loss(view1, view2):
        loss = (2 - 2*F.cosine_similarity(view1, view2)).mean()
        return loss

    def get_cluster_center(self, z1, z2, dis, predict_labels):
        high_confidence = torch.min(dis, dim=1).values
        max_dis_idx=int(len(high_confidence) * self.threshold)
        max_dis = torch.sort(high_confidence).values[max_dis_idx]
        high_confidence_idx = np.argwhere(high_confidence < max_dis)[0]
        y_sam = torch.tensor(predict_labels)[high_confidence_idx]
        all_class=np.unique(predict_labels).tolist()
        sampled_class=torch.unique(y_sam).numpy().tolist()
        if len(all_class)>len(sampled_class):
            for t_class in all_class:
                if t_class not in sampled_class:
                    t_ids=np.where(predict_labels==t_class)[0]
                    t_dis=dis[t_ids, t_class]
                    arg=np.argsort(t_dis)
                    cnt=int(len(arg) * self.threshold)
                    if cnt==0:
                        cnt+=1
                    arg=arg[:cnt]
                    new_sampled=t_ids[arg]
                    high_confidence_idx=torch.concat([high_confidence_idx, torch.tensor(new_sampled)])
            high_confidence_idx=torch.sort(high_confidence_idx).values
            y_sam = torch.tensor(predict_labels)[high_confidence_idx]
        index = torch.tensor(range(self.n_sp))[high_confidence_idx]
        index=index.to(self.device)
        y_sam=y_sam.to(self.device)
        index = index[torch.argsort(y_sam)]
        class_num = {}
        for label in torch.sort(y_sam).values:
            label = label.item()
            if label in class_num.keys():
                class_num[label] += 1
            else:
                class_num[label] = 1
        key = sorted(class_num.keys())
        centers_1 = torch.tensor([], device=self.device)
        centers_2 = torch.tensor([], device=self.device)
        st = 0
        for i in range(len(key)):
            now = index[st:st + class_num[key[i]]]
            st += class_num[key[i]]
            centers_1 = torch.cat(
                [centers_1, torch.mean(z1[now], dim=0).unsqueeze(0)], dim=0)
            centers_2 = torch.cat(
                [centers_2, torch.mean(z2[now], dim=0).unsqueeze(0)], dim=0)
        centers_1 = F.normalize(centers_1, dim=1, p=2)
        centers_2 = F.normalize(centers_2, dim=1, p=2)
        return centers_1, centers_2

    def prototype_contrastive_loss(self, centers_1, centers_2, predict_labels):
        d_q = centers_1.mm(centers_1.T) / self.temperature
        d_k = (centers_1 * centers_2).sum(dim=1) / self.temperature
        d_q = d_q.float()
        d_q[torch.arange(self.cluster_num), torch.arange(self.cluster_num)] = d_k

        predict_labels = torch.from_numpy(predict_labels).long().to(self.device)
        zero_classes = torch.arange(self.cluster_num).to(self.device)[torch.sum(
            F.one_hot(torch.unique(predict_labels), self.cluster_num), dim=0) == 0]
        mask = torch.zeros((self.cluster_num, self.cluster_num),
                           dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.cluster_num, self.cluster_num))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.cluster_num - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.cluster_num, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.cluster_num - len(zero_classes))
        return loss
    
    def moco_infoNCE(self, q, k, criterion, temp=0.1):
        logits=torch.matmul(q, k.T)/temp
        labels = torch.arange(q.shape[0]).to(self.device)
        loss = criterion(logits, labels)
        return loss

    def train_forward(self, sp_feat, aug_feat, adj):
        sp = F.relu(self.gc1(sp_feat, adj))
        sp = F.dropout(sp, self.dropout, training=self.training)
        sp = F.relu(self.gc2(sp, adj))
        sp = F.dropout(sp, self.dropout, training=self.training)
        sp_out1 = self.gc3_1(sp, adj)
        sp_out2 = self.gc3_2(sp, adj)
        
        aug = F.relu(self.gc1(aug_feat, adj))
        aug = F.dropout(aug, self.dropout, training=self.training)
        aug = F.relu(self.gc2(aug, adj))
        aug = F.dropout(aug, self.dropout, training=self.training)
        aug_out1 = self.gc3_1(aug, adj)
        aug_out2 = self.gc3_2(aug, adj)

        return sp_out1, sp_out2, aug_out1, aug_out2

    def eval_forward(self, sp_feat, adj):
        x = F.relu(self.gc1(sp_feat, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        out1 = self.gc3_1(x, adj)
        out2 = self.gc3_2(x, adj)

        return out1, out2