##################################################################EXAMPLE##############################################################
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from shapelet import MultiTree
import random
import tensorflow as tf

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
test_true=[str(1)]*3+[str(2)]*5+[str(3)]*4


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
plt.title('Class 3: cosine and uniform', fontsize=12)

plt.tight_layout()
plt.show()

trial_series=class1_series+class2_series+class3_series
ori_label=[str(1)]*20+[str(2)]*22+[str(3)]*21

mt=MultiTree(MAXLEN=16,MINLEN=3,max_depth=3,min_samples_split=5)
with tf.device('/GPU:0'):
    mt.fit(trial_series, ori_label)
structure=mt.get_tree_structure()
y_pred=mt.predict(test_series)

cm=confusion_matrix(test_true, y_pred.tolist())
print(cm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

for ts in test_series:
    plt.plot(ts, color='blue', alpha=0.6)
plt.plot(structure["feature"], color='red', alpha=0.6)
plt.plot(structure["sub_nodes"][0]["feature"], color='red', alpha=0.6)

for ts in trial_series:
    plt.plot(ts, color='blue', alpha=0.6)
plt.plot(structure["feature"], color="red",alpha=0.6)
plt.plot(structure["sub_nodes"][0]["feature"], color="red",alpha=0.6)
