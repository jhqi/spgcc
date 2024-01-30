import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, adjusted_rand_score
import sys
from scipy.optimize import linear_sum_assignment


def Hungarian(A):
    _, col_ind = linear_sum_assignment(A)
    # Cost can be found as A[row_ind, col_ind].sum()
    return col_ind

def BestMap(L1, L2):
    L1 = L1.flatten(order='F').astype(float)
    L2 = L2.flatten(order='F').astype(float)
    if L1.size != L2.size:
        sys.exit('size(L1) must == size(L2)')
    Label1 = np.unique(L1)
    nClass1 = Label1.size
    Label2 = np.unique(L2)
    nClass2 = Label2.size
    nClass = max(nClass1, nClass2)

    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass]).astype(float)
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    c = Hungarian(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(0, nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]

    # ori2mapped=[0]*nClass
    # for i in range(0, nClass2):
    #     newL2[L2 == Label2[i]] = Label1[c[i]]
    #     ori_label=int(Label2[i])
    #     mapped_label=int(Label1[c[i]])
    #     ori2mapped[ori_label]=mapped_label
    # ori2mapped=np.array(ori2mapped, dtype=np.int32)

    return newL2

def cal_metric(label_gt, label_predict, n_class, is_refined=False):
    if not is_refined:
        label_predict=BestMap(label_gt,label_predict)

    #混淆矩阵
    #col_name是实际类别
    #row_name是预测类别
    confuse_matrix=np.zeros((n_class,n_class),dtype=np.int64)
    for i in range(n_class):
        for j in range(n_class):
            #confuse_matrix[i,j]表示预测为i，实际为j的样本数量
            predict_i=(label_predict==i).astype(np.int64)
            gt_j=(label_gt==j).astype(np.int64)
            confuse_matrix[i,j]=np.sum((predict_i+gt_j)==2)
    
    OA=np.sum(np.diagonal(confuse_matrix))/np.sum(confuse_matrix)

    #kappa
    n_gt=np.sum(confuse_matrix,axis=0) #实际每一类的样本数
    n_pred=np.sum(confuse_matrix,axis=1) #预测每一类的样本数
    n=np.sum(n_gt) #总样本数
    pe=np.dot(n_gt,n_pred)/(n*n)
    po=OA
    KAPPA=(po-pe)/(1-pe)

    #producer's acc
    PA=np.diagonal(confuse_matrix)/n_gt

    #NMI
    NMI=metrics.normalized_mutual_info_score(label_gt, label_predict)
    return OA, KAPPA, NMI, PA


def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def cal_f_p_r(T, H):
    #T: label_gt
    #H: label_predict
    N = len(T)
    numT = 0
    numH = 0
    numI = 0
    for n in range(N):
        Tn = T[n+1:]==T[n]
        Hn = H[n+1:]==H[n]
        numT += Tn.sum()
        numH += Hn.sum()
        numI += (Tn*Hn).sum()

    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p+r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)

    return f, p, r


def full_metric(label_gt, label_predict, is_refined=False):
    if not is_refined:
        label_predict=BestMap(label_gt, label_predict)
    M=confusion_matrix(label_gt, label_predict) #M[i,j]表示真实i类，预测j类的样本个数
    OA=np.sum(np.diagonal(M))/np.sum(M)
    AA=np.mean(np.diagonal(M)/np.sum(M, axis=1))
    KAPPA=cohen_kappa_score(label_gt, label_predict)
    NMI=metrics.normalized_mutual_info_score(label_gt, label_predict)
    ARI=adjusted_rand_score(label_gt, label_predict)
    F1, PRECISION, RECALL=cal_f_p_r(label_gt, label_predict)
    PURITY=purity_score(label_gt, label_predict)
    return OA, AA, KAPPA, NMI, ARI, F1, PRECISION, RECALL, PURITY