import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random
import tensorflow as tf
from statistics import mode
# from sklearn.metrics import mutual_info_score

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
    key["dist"]=v1_values
    key["D1"]=[index for index,value in enumerate(v1_values) if value <= split_point]
    key["D2"]=[index for index,value in enumerate(v1_values) if value > split_point]
    return key

def check_candidate(D, S, original_label,seed, Metric):
    objects_histogram = []
    for T in D:
        dist = subsequence_dist(T, S, Metric)
        objects_histogram.append(dist)
    return CalculateInformationGain(objects_histogram,original_label,seed)

def find_best_shapelet(D, MAXLEN, MINLEN, original_label, Metric="euclidean",seed=1):
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
                s_dist=check["dist"]
                D1=check["D1"]
                D2=check["D2"]
    return {"best_shapelet":bsf_shapelet,"best_gain":bsf_gain,
            "original_entropy":max_gain,"split_point":sp,
            "shapelet_dist":s_dist,"D1":D1,"D2":D2}

def majority_vote(y):
    return mode(y)

class MultiTree:
    def __init__(self, MAXLEN, MINLEN, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.maxlen=MAXLEN
        self.minlen=MINLEN

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'class': majority_vote(y)}  

        if len(np.unique(y)) == 1:
            return {'class': y[0]}
        
        res=find_best_shapelet(X, self.maxlen, self.minlen, y, Metric="euclidean",seed=1)

        best_feature=res["best_shapelet"]
        best_thresholds=res["split_point"]
        
        ds1=[X[i] for i in res["D1"]]
        label1=[y[i] for i in res["D1"]]
        sub1={"X":ds1,"y":label1}
        ds2=[X[i] for i in res["D2"]]
        label2=[y[i] for i in res["D2"]]
        sub2={"X":ds2,"y":label2}       
        subsets=[sub1,sub2] 

        if best_feature is None:
            return {'class': majority_vote(y)}

        node = {
            'feature': best_feature,        
            'thresholds': best_thresholds,  
            'sub_nodes': []                 
        }

        for subset in subsets:
            node['sub_nodes'].append(
                self._build_tree(subset['X'], subset['y'], depth+1)  
            )
        return node
    def print_tree_structure(self, node=None, depth=0):
        if node is None:
            node = self.tree  
        prefix = "  " * depth  

        if 'class' in node:
            print(f"{prefix}Leaf: Predicted Class = {node['class']}")
            return

        feature = node['feature']
        thresholds = node['thresholds']
        print(f"{prefix}Depth={depth}: Split on Feature {feature}, Thresholds={thresholds}")

        for i, sub_node in enumerate(node['sub_nodes']):
            print(f"{prefix}  Sub-node {i}:")
            self.print_tree_structure(sub_node, depth + 1)

    def get_tree_structure(self):
        return self._traverse_tree(self.tree)

    def _traverse_tree(self, node):
        if 'class' in node:  
            return {
                'type': 'leaf',
                'class': node['class']
            }
        else:  
            return {
                'type': 'internal',
                'feature': list(node['feature']),
                'thresholds': node['thresholds'],
                'sub_nodes': [self._traverse_tree(sub) for sub in node['sub_nodes']]
            }
            
    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])
    def _predict_single(self, x, node):
        if 'class' in node:
            return node['class']
        feature_value = subsequence_dist(x, node["feature"], metric="euclidean")

        threshold = node['thresholds'] 
        if feature_value <= threshold:
            return self._predict_single(x, node['sub_nodes'][0])
        else:
            return self._predict_single(x, node['sub_nodes'][1])

##################################################################EXAMPLE##############################################################
import matplotlib.pyplot as plt
np.random.seed(13)

def generate_class1(length):
    x = np.linspace(0, 4 * np.pi, length)
    return 5 * np.sin(x) + np.random.normal(0, 0.5, length)

def generate_class2(length):
    trend = np.linspace(0, 5+10, length)
    return trend + np.random.normal(0, 0.3, length)

def generate_class3(length):
    x = np.linspace(0, 4 * np.pi, length)
    trend = random.uniform(-3, 3)
    return  np.cos(x) + trend + np.random.normal(0, 0.3, length)

class1_series = [generate_class1(np.random.randint(40, 61)) for _ in range(20)]
class2_series = [generate_class2(np.random.randint(40, 61)) for _ in range(22)]
class3_series = [generate_class3(np.random.randint(40, 61)) for _ in range(21)]

np.random.seed(41)
test_series=[generate_class1(np.random.randint(40, 61)) for _ in range(3)]+[generate_class2(np.random.randint(40, 61)) for _ in range(5)]+[generate_class3(np.random.randint(40, 61)) for _ in range(4)]

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
for ts in class1_series:
    plt.plot(ts, color='blue', alpha=0.6)
plt.title('Class 1: sine', fontsize=12)

plt.subplot(3, 1, 2)
for ts in class2_series:
    plt.plot(ts, color='red', alpha=0.6)
plt.title('Class 2: linear', fontsize=12)

plt.subplot(3, 1, 3)
for ts in class3_series:
    plt.plot(ts, color='green', alpha=0.6)
plt.title('Class 2: linear', fontsize=12)

plt.tight_layout()
plt.show()

trial_series=class1_series+class2_series+class3_series
ori_label=[str(1)]*20+[str(2)]*22+[str(3)]*21

mt=MultiTree(MAXLEN=16,MINLEN=3,max_depth=3,min_samples_split=5)
with tf.device('/GPU:0'):
    mt.fit(trial_series, ori_label)
structure=mt.get_tree_structure()
y_pred=mt.predict(test_series)

for ts in test_series:
    plt.plot(ts, color='blue', alpha=0.6)
plt.plot(structure["feature"], color='red', alpha=0.6)
plt.plot(structure["sub_nodes"][0]["feature"], color='red', alpha=0.6)

for ts in trial_series:
    plt.plot(ts, color='blue', alpha=0.6)
plt.plot(structure["feature"], color="red",alpha=0.6)
plt.plot(structure["sub_nodes"][0]["feature"], color="red",alpha=0.6)
