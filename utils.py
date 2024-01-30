import torch
from torch_clustering import PyTorchKMeans
from torch.utils.data.dataset import Dataset
import random
import numpy as np
from cal_metric import full_metric
import scipy.sparse as sp
import scipy.io as sio

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class AEDataset(Dataset):
    def __init__(self, Datapath, transform):
        self.Datalist = np.load(Datapath)
        self.transform = transform

    def __getitem__(self, index):
        Data = self.transform(self.Datalist[index].astype('float64'))
        Data = Data.view(1, Data.shape[0], Data.shape[1], Data.shape[2])
        return Data

    def __len__(self):
        return len(self.Datalist)


def read_mat(filename):
    mat = sio.loadmat(filename)
    keys = [k for k in mat.keys() if k != '__version__' and k != '__header__' and k != '__globals__']
    arr = mat[keys[0]]
    return arr


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def my_clustering_gpu(feature, true_labels, cluster_num, labeled_p_in_sp):
    kmeans = PyTorchKMeans(metric='euclidean', init='k-means++', n_clusters=cluster_num, n_init=15, verbose=False)
    predict_labels = kmeans.fit_predict(feature)
    center = kmeans.cluster_centers_
    dis=torch.cdist(feature, center, p=2)
    dis=dis.cpu()
    predict_labels = predict_labels.cpu().numpy()
    pixel_pred = predict_labels[labeled_p_in_sp]
    OA, AA, KAPPA, NMI, ARI, F1, PRECISION, RECALL, PURITY = full_metric(true_labels, pixel_pred, is_refined=False)
    return 100*OA, 100*AA, 100*KAPPA, 100*NMI, 100*ARI, 100*F1, 100*PRECISION, 100*RECALL, 100*PURITY, predict_labels, dis

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def process_adj(adj, norm='sym', renorm=True):
    # adj: numpy dense matrix
    adj = adj-np.diag(np.diag(adj))
    adj = sp.csr_matrix(adj)
    adj.eliminate_zeros()
    adj = sp.coo_matrix(adj)
    if renorm:
        adj = adj + sp.eye(adj.shape[0])
    # degree vector
    rowsum = np.array(adj.sum(1))
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(
            degree_mat_inv_sqrt).tocoo()
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj).tocoo()
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
    return adj_normalized

def random_sample_pixel_from_sp(p_in_sp_list, cnn_encoding):
    sampled_encoding=[]
    n_sp=len(p_in_sp_list)
    for i in range(n_sp):
        ids=p_in_sp_list[i]
        sampled_idx=int(np.random.choice(ids, size=1))
        sampled_encoding.append(cnn_encoding[sampled_idx,:]*len(ids))
    sampled_encoding=np.stack(sampled_encoding, axis=0)
    return sampled_encoding