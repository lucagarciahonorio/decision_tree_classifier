from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import math
from collections import Counter

# 1. Carregar os dados
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 2. Separar em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Modelo com Scikit-learn
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_sklearn = clf.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Acurácia (Scikit-learn): {acc_sklearn:.2f}")

# 4. Implementação básica de uma Árvore de Decisão

def entropy(y):
    counts = Counter(y)
    probabilities = [v / len(y) for v in counts.values()]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def split_dataset(X, y, feature, threshold):
    left_X, left_y = [], []
    right_X, right_y = [], []
    for xi, yi in zip(X, y):
        if xi[feature] < threshold:
            left_X.append(xi)
            left_y.append(yi)
        else:
            right_X.append(xi)
            right_y.append(yi)
    return (left_X, left_y), (right_X, right_y)

def best_split(X, y):
    best_gain = -1
    best_feature, best_threshold = None, None
    base_entropy = entropy(y)
    n_features = len(X[0])
    for feature in range(n_features):
        thresholds = set([x[feature] for x in X])
        for threshold in thresholds:
            (left_X, left_y), (right_X, right_y) = split_dataset(X, y, feature, threshold)
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            gain = base_entropy - (len(left_y)/len(y))*entropy(left_y) - (len(right_y)/len(y))*entropy(right_y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=3):
    if len(set(y)) == 1 or depth == max_depth:
        return Node(value=Counter(y).most_common(1)[0][0])
    feature, threshold = best_split(X, y)
    if feature is None:
        return Node(value=Counter(y).most_common(1)[0][0])
    (left_X, left_y), (right_X, right_y) = split_dataset(X, y, feature, threshold)
    left_node = build_tree(left_X, left_y, depth+1, max_depth)
    right_node = build_tree(right_X, right_y, depth+1, max_depth)
    return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

def predict_tree(x, node):
    if node.value is not None:
        return node.value
    if x[node.feature] < node.threshold:
        return predict_tree(x, node.left)
    else:
        return predict_tree(x, node.right)

# 5. Treinar a árvore feita na mão
X_train_list = X_train.tolist()
X_test_list = X_test.tolist()
tree_root = build_tree(X_train_list, y_train.tolist(), max_depth=3)
y_pred_custom = [predict_tree(x, tree_root) for x in X_test_list]
acc_custom = accuracy_score(y_test, y_pred_custom)
print(f"Acurácia (Árvore feita na mão): {acc_custom:.2f}")
