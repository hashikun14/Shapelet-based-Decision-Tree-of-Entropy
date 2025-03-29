import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import random
def generate_candidates(T, MAXLEN, MINLEN):
    pool = []
    l = MAXLEN
    while l >= MINLEN:
        for i in range(0,len(T) - l + 1,1):
            subsequence = tuple(T[i:i + l]) 
            pool.append(subsequence)  
        l -= 1    
    return pool

def subsequence_dist(T, S, metric):
    T, S = np.array(T), np.array(S)
    len_T, len_S = len(T),len(S)
    min_dist = np.inf
    for i in range(len_T - len_S + 1):
        window = T[i:i + len_S]
        if metric=="euclidean":
            sum_sq = np.sum((window - S)**2)
        if metric=="L1":
            sum_sq = np.sum(np.abs(window - S))
        if sum_sq < min_dist:
            min_dist = sum_sq
    return np.sqrt(min_dist)

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities))

def CalculateInformationGain(obj_hist,original_label,seed):
    key={}
    v1_values=np.array(obj_hist)
    clf = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=seed)
    clf.fit(v1_values.reshape(-1, 1),original_label)
    split_point=clf.tree_.threshold[0]
    tree = clf.tree_
    H_root = tree.impurity[0]
    H_left = tree.impurity[1]
    H_right = tree.impurity[2]
    w_left = tree.n_node_samples[1] / tree.n_node_samples[0]
    w_right = tree.n_node_samples[2] / tree.n_node_samples[0]
    
    info_gain=H_root-(w_left*H_left+w_right*H_right)
    key["split_point"]=split_point
    key["gain"]=info_gain
    return key

def check_candidate(D, S, original_label,seed, Metric):
    objects_histogram = []
    for T in D:
        dist = subsequence_dist(T, S, Metric)
        objects_histogram.append(dist)
    return CalculateInformationGain(objects_histogram,original_label,seed)

def find_best_shapelet(D, MAXLEN, MINLEN, original_label, Metric="euclidean",seed=1):
    candidates=generate_candidates(D, MAXLEN, MINLEN)
    candidates.sort(key=lambda x: len(x))
    
    bsf_gain = 0
    bsf_shapelet = None  
    sp=0
    max_gain=entropy(original_label)
    for T in D:
        candidates=generate_candidates(T, MAXLEN, MINLEN)
        candidates.sort(key=lambda x: len(x))
        for S in candidates:
            check = check_candidate(D, S, original_label, seed, Metric)
            gain=check["gain"]
            if gain > bsf_gain:
                bsf_gain = gain
                bsf_shapelet = S
                sp=check["split_point"]
            if bsf_gain==max_gain:
                break    
    return {"best_shapelet":bsf_shapelet,"best_gain":bsf_gain,"original_entropy":max_gain,"split_point":sp}

##################################################################EXAMPLE##############################################################
import matplotlib.pyplot as plt
np.random.seed(13)

def generate_class1(length):
    x = np.linspace(0, 4 * np.pi, length)
    return 1.25 * np.sin(x) + np.random.normal(0, 0.5, length)

def generate_class2(length):
    trend = np.linspace(0, 5, length)
    return trend + np.random.normal(0, 0.3, length)

def generate_class3(length):
    x = np.linspace(0, 4 * np.pi, length)
    trend = np.linspace(0, 5, length)
    return 2 * np.sin(x) + trend + np.random.normal(0, 0.3, length)

class1_series = [generate_class1(np.random.randint(40, 61)) for _ in range(10)]
class2_series = [generate_class3(np.random.randint(40, 61)) for _ in range(12)]

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
for ts in class1_series:
    plt.plot(ts, color='blue', alpha=0.6)
plt.title('Class 1: sine', fontsize=12)

plt.subplot(2, 1, 2)
for ts in class2_series:
    plt.plot(ts, color='red', alpha=0.6)
plt.title('Class 2: linear', fontsize=12)

plt.tight_layout()
plt.show()

trial_series=class1_series+class2_series
ori_label=[str(1)]*10+[str(2)]*12
trial_df=pd.DataFrame(zip(trial_series,ori_label))
trial_df=trial_df.sample(frac=1,random_state=590).reset_index(drop=True)
buon=find_best_shapelet(trial_series, 32, 16,ori_label,Metric="L1")
plt.plot(buon["best_shapelet"],color="black")
print(buon["split_point"])

# pp=generate_candidates(trial_series, 35, 15)
# T=trial_series[0]
# S=pp[244]
# tri_dist=subsequence_dist(T,S)
# obj_hist=objects_histogram
# original_label=ori_label
# subsequence_dist(T=T, S=S, metric=Metric)
# D=trial_series
# seed=1
# np.random.seed(123)
# indice=set(random.sample(range(179), 120))
# covid_=[]
# for i in range(covid21.shape[0]):
#     covid_.append(np.array(covid21.iloc[i,1:556],dtype="float32"))
# train=[covid_[i] for i in indice]
# test=[covid_[i] for i in set(range(179))-indice]
# train_label=[covid21.iloc[i,-1] for i in indice]
# test_label=[covid21.iloc[i,-1] for i in set(range(179))-indice]
# bene=find_best_shapelet(train, 250, 200, train_label)

