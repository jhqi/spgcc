import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from Preprocess import load_label
from utils import process_adj, setup_seed, my_clustering_gpu, read_mat, random_sample_pixel_from_sp
from models import SPGCC
import pandas as pd
import warnings
import os
import sys

warnings.filterwarnings("ignore")
run_dir = os.path.dirname(sys.argv[0])
if run_dir == "":
    run_dir = "."
os.chdir(run_dir)

device = torch.device("cuda:3")

dataset_name = "PU"
cluster_num = 9
max_no_better = 150
epochs = 500

param = {
    "n_sp": 2200,
    "lr": 1e-4,
    "threshold": 0.25,
    "temperature": 0.5,
    "alpha": 0.1,
    "dropout": 0.1,
}

n_sp = param["n_sp"]
lr = param["lr"]
threshold = param["threshold"]
temperature = param["temperature"]
alpha = param["alpha"]
dropout = param["dropout"]

# prepare label
label_gt, labeled_p_in_sp = load_label(dataset_name, n_sp=n_sp)

# pixel cnn encoding
cnn_encoding = np.load("pretrained_emb/PU_pretrained_emb.npy")

# sp_map
sp_map = read_mat("sp_seg_map/PU_sp_map_2200.mat").astype(np.int32)
sp_map = sp_map.reshape(-1)

# p_in_sp_list
p_in_sp_list = []
for i in range(n_sp):
    ids = np.where(sp_map == i)[0]
    p_in_sp_list.append(ids)

# sp encoding
N = cnn_encoding.shape[0]
pixel_in_sp = np.zeros((N, n_sp), dtype=np.float32)
for i in range(N):
    pixel_in_sp[i, sp_map[i]] = 1
sp_feat_np = np.matmul(pixel_in_sp.T, cnn_encoding)
sp_feat = torch.from_numpy(sp_feat_np).to(device)

# superpixel adj
adj = np.load("sp_adj/PU_sp_adj_2200.npy")
adj = process_adj(adj)
adj = adj.to(device)

for seed in range(10):
    # fix seed
    setup_seed(seed)
    print(f"seed:{seed}")

    # model
    model = SPGCC(
        nfeat=cnn_encoding.shape[1],
        nhid=1024,
        nout=512,
        n_sp=n_sp,
        cluster_num=cluster_num,
        dropout=dropout,
        threshold=threshold,
        temperature=temperature,
        alpha=alpha,
        device=device,
    ).to(device)

    # optim
    optimizer = Adam(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    best_oa = 0
    best_aa = 0
    best_kappa = 0
    best_nmi = 0
    best_ari = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_purity = 0
    best_metric = 0

    # init metric
    oa, aa, kappa, nmi, ari, f1, precision, recall, purity, predict_labels, dis = my_clustering_gpu(
        sp_feat, label_gt, cluster_num, labeled_p_in_sp
    )

    # train
    no_better = 0
    for epoch in range(1, epochs + 1):
        aug_feat = random_sample_pixel_from_sp(p_in_sp_list, cnn_encoding)
        aug_feat = torch.from_numpy(aug_feat).to(device)

        model.train()
        optimizer.zero_grad()
        sp_out1, sp_out2, aug_out1, aug_out2 = model.train_forward(sp_feat, aug_feat, adj)  # sp, aug

        if epoch <= 20:
            z1 = F.normalize(sp_out1, dim=1, p=2)
            z2 = F.normalize(sp_out2, dim=1, p=2)
            loss = model.moco_infoNCE(z1, z2, criterion)
            loss.backward()
            optimizer.step()
            print(f"epoch:{epoch}, graph_pretrain_loss:{loss.item()}")
        else:
            align_loss1 = model.neighbour_alignment_loss(sp_out1, sp_out2)
            align_loss2 = model.neighbour_alignment_loss(aug_out1, aug_out2)
            align_loss3 = model.neighbour_alignment_loss(sp_out1, aug_out1)
            align_loss4 = model.neighbour_alignment_loss(sp_out1, aug_out2)
            align_loss5 = model.neighbour_alignment_loss(sp_out2, aug_out1)
            align_loss6 = model.neighbour_alignment_loss(sp_out2, aug_out2)
            align_loss = (align_loss1 + align_loss2 + align_loss3 + align_loss4 + align_loss5 + align_loss6) / 6

            z1 = F.normalize(sp_out1, dim=1, p=2)
            z2 = F.normalize(sp_out2, dim=1, p=2)
            centers_1, centers_2 = model.get_cluster_center(z1, z2, dis, predict_labels)
            contra_loss1 = model.prototype_contrastive_loss(centers_1, centers_2, predict_labels)
            contra_loss2 = model.prototype_contrastive_loss(centers_2, centers_1, predict_labels)
            contra_loss = (contra_loss1 + contra_loss2) / 2

            loss = align_loss + model.alpha * contra_loss
            loss.backward()
            optimizer.step()
            print(
                f"epoch:{epoch}, loss:{loss.item()}, align_loss:{align_loss.item()}, contra_loss:{contra_loss.item()}"
            )

        if epoch % 5 == 0:
            model.eval()
            with torch.set_grad_enabled(False):
                out1, out2 = model.eval_forward(sp_feat, adj)
            z1 = F.normalize(out1, dim=1, p=2)
            z2 = F.normalize(out2, dim=1, p=2)
            hidden_emb = torch.concat([z1, z2], dim=1)
            oa, aa, kappa, nmi, ari, f1, precision, recall, purity, predict_labels, dis = my_clustering_gpu(
                hidden_emb, label_gt, cluster_num, labeled_p_in_sp
            )
            metric = oa + aa + nmi + ari + f1
            if metric >= best_metric:
                best_oa = oa
                best_aa = aa
                best_kappa = kappa
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_purity = purity
                best_metric = metric
                no_better = 0
            else:
                no_better += 5
            print(
                f"epoch:{epoch}, oa:{oa}, aa:{aa}, kappa:{kappa}, nmi:{nmi}, ari:{ari}, f1:{f1}, precision:{precision}, recall:{recall}, purity:{purity}, best oa:{best_oa}, best_aa:{best_aa}, best kappa:{best_kappa}, best nmi:{best_nmi}, best_ari:{best_ari}, best_f1:{best_f1}, best_precision:{best_precision}, best_recall:{best_recall}, best_purity:{best_purity}"
            )
            if no_better >= max_no_better:
                break

    print(f"best oa:{best_oa}")
    print(f"best aa:{best_aa}")
    print(f"best kappa:{best_kappa}")
    print(f"best nmi:{best_nmi}")
    print(f"best ari:{best_ari}")
    print(f"best f1:{best_f1}")
    print(f"best precision:{best_precision}")
    print(f"best recall:{best_recall}")
    print(f"best purity:{best_purity}")
    dic = {
        "oa": [best_oa],
        "aa": [best_aa],
        "kappa": [best_kappa],
        "nmi": [best_nmi],
        "ari": [best_ari],
        "f1": [best_f1],
        "precision": [best_precision],
        "recall": [best_recall],
        "purity": [best_purity],
    }
    df = pd.DataFrame(dic)

    filename = f"{dataset_name}_res.csv"

    if not os.path.exists(filename):
        df.to_csv(filename, index=False, header=True, encoding="utf-8")
    else:
        df.to_csv(filename, mode="a", index=False, header=False, encoding="utf-8")
