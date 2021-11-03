# coding:utf8

import sys
from random import randint
from random import randrange
from collections import Counter

import numpy as np

sys.path.append('..')
# hyper parameter
# fold size (% of dataset size) e.g. 3 means 30%
FOLD_SIZE = 10
# number of trees
N_TREES = 20
# max tree depth
MAX_DEPTH = 30
# min size of tree node
MIN_NODE = 1


class DecisionTree:
    def __init__(self, max_depth, min_node_size):
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.final_tree = {}

    def calculate_gini(self, child_nodes):
        """
        child nodes
        :param child_nodes: [[left_nodes], [right_nodes]]
        :return:
        """
        n = 0
        for node in child_nodes:
            n = n + len(node)  #

        gini = 0
        for node in child_nodes:
            m = len(node)  # 某一切分数组的总数
            if m == 0:
                continue
            y = []
            for row in node:
                y.append(row[-1])  # target分布

            freq = Counter(y).values()
            node_gini = 1
            for i in freq:
                node_gini = node_gini - (i / m) ** 2  # 1 - p**2
            gini = gini + (m / n) * node_gini  # pD * g(D, Ai)
        return gini

    def apply_split(self, feature_index, threshold, data):
        instances = data.tolist()
        left_child = []
        right_child = []
        for row in instances:
            if row[feature_index] < threshold:
                left_child.append(row)
            else:
                right_child.append(row)
        left_child = np.array(left_child)
        right_child = np.array(right_child)
        return left_child, right_child

    def find_best_split(self, data):
        num_of_features = len(data[0]) - 1
        gini_scores = 1000
        feature_index = 0
        feature_value = 0

        for column in range(num_of_features):
            for row in data:
                value = row[column]
                left, right = self.apply_split(column, value, data)
                children = [left, right]
                scores = self.calculate_gini(child_nodes=children)
                if scores < gini_scores:
                    feature_index = column
                    feature_value = value
                    gini_scores = scores
                    child_nodes = children

        node = {"feature": feature_index, "value": feature_value, "children": child_nodes}
        return node

    def calc_class(self, node):
        y = []
        for row in node:
            y.append(row[-1])
        count_y = Counter(y)
        return count_y.most_common(1)[0][0]

    def recursive_split(self, node, depth):
        l, r = node["children"]
        del node["children"]
        if l.size == 0:
            c_value = self.calc_class(r)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        elif r.size == 0:
            c_value = self.calc_class(l)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        # Check if tree has reached max depth
        if depth >= self.max_depth:
            # Terminate left child node
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
            # Terminate right child node
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
            return
        # process left child
        if len(l) <= self.min_node_size:
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
        else:
            node["left"] = self.find_best_split(l)
            self.recursive_split(node["left"], depth + 1)
        # process right child
        if len(r) <= self.min_node_size:
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
        else:
            node["right"] = self.find_best_split(r)
            self.recursive_split(node["right"], depth + 1)

    """
        Apply the recursive split algorithm on the data in order to build the decision tree
        Parameters:
        X (np.array): Training data

        Returns tree (dict): The decision tree in the form of a dictionary.
    """

    def train(self, X):
        # Create initial node
        tree = self.find_best_split(X)
        # Generate the rest of the tree via recursion
        self.recursive_split(tree, 1)
        self.final_tree = tree
        return tree

    """
        Prints out the decision tree.
        Parameters:
        tree (dict): Decision tree

    """

    def print_dt(self, tree, depth=0):
        if "feature" in tree:
            print(
                "\nSPLIT NODE: feature #{} < {} depth:{}\n".format(
                    tree["feature"], tree["value"], depth
                )
            )
            self.print_dt(tree["left"], depth + 1)
            self.print_dt(tree["right"], depth + 1)
        else:
            print(
                "TERMINAL NODE: class value:{} depth:{}".format(
                    tree["class_value"], tree["depth"]
                )
            )

    def predict_single(self, tree, instance):
        if not tree:
            print("ERROR: Please train the decision tree first")
            return -1
        if "feature" in tree:
            if instance[tree["feature"]] < tree["value"]:
                return self.predict_single(tree["left"], instance)
            else:
                return self.predict_single(tree["right"], instance)
        else:
            return tree["class_value"]

    def predict(self, X):
        y_predict = []
        for row in X:
            y_predict.append(self.predict_single(self.final_tree, row))
        return np.array(y_predict)


class RandomForest:
    def __init__(self, n_trees, fold_size):
        self.n_trees = n_trees
        self.fold_size = fold_size
        self.trees = list()

    def cross_validation_split(self, dataset, n_folds, p):
        """
        This function splits the given dataset into n-folds with replacement
        :param dataset:  np array of the given dataset
        :param n_folds: number of folds in which the dataset should be split. Must be equal to the number of trees the user wants to train
        :param p: suggests the percentage of the dataset's size the size of a single fold should be.
        :return: list with the k-folds
        """
        dataset_split = list()
        fold_size = int(len(dataset) * p / 10)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset))
                fold.append(dataset[index])
            fold_set = np.array(fold)
            dataset_split.append(fold_set)
        return dataset_split

    def random_features(self, splits):
        """
        randomizes the selection of the features each tree will be trained on.
        :param splits: list of folds
        :return: list with the k-folds with some features randomly removed
        """
        dataset_split = list()
        l = len(splits[0][0])  # 第一个分组的第一个元素的长度就是特征的数量
        n_features = int((l - 1) * 5 / 10)  # 随机选择一半数量的特征
        for split in splits:
            for i in range(n_features):  # 随机删除一个特征
                rng = list(range(len(split[0]) - 1))
                selected = rng.pop(randint(0, len(rng) - 1))
                split = np.delete(split, selected, 1)
            set = np.array(split)
            dataset_split.append(set)  # 每个分组的数据都随机删除一半的特征
        return dataset_split

    def print_tree(self):
        i = 1
        for t in self.trees:
            print("Tree#", i)
            temp = t.final_tree
            t.print_dt(temp)
            print("\n")
            i = i + 1

    def train(self, X):
        train_x = self.cross_validation_split(X, self.n_trees, self.fold_size)
        train_x = self.random_features(train_x)
        for fold in train_x:
            dt = DecisionTree(MAX_DEPTH, MIN_NODE)
            dt.train(fold)
            self.trees.append(dt)

    def predict(self, X):
        predicts = list()
        final_predicts = list()
        for tree in self.trees:
            predicts.append(tree.predict(X))
        for i in range(len(predicts[0])):
            values = list()
            for j in range(len(predicts)):
                values.append(predicts[j][i])
            final_predicts.append(max(set(values), key=values.count))
        return final_predicts, predicts


if __name__ == "__main__":

    # Training data
    train_data = np.loadtxt("example_data/data.txt", delimiter=",")
    train_y = np.loadtxt("example_data/targets.txt")

    mock_train = np.loadtxt("example_data/mock_data.csv", delimiter=",")
    mock_y = mock_train[:, -1]

    # Build and train model
    rf = RandomForest(N_TREES, FOLD_SIZE)
    rf.train(mock_train)

    # Evaluate model on training data
    y_pred, y_pred_ind = rf.predict(mock_train)
    print(f"Accuracy of random forest: {sum(y_pred == mock_y) / mock_y.shape[0]}")
    print("\nAccuracy for each individual tree:")
    c = 1
    for i in y_pred_ind:
        print("\nTree", c)
        print(f"Accuracy: {sum(i == mock_y) / mock_y.shape[0]}")
        c = c + 1
