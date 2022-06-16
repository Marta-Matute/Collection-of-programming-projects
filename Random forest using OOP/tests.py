import logging
import math
from sys import stdout, stderr
from time import time

import numpy as np
import sklearn.datasets

from criterion import Gini
from dataset import Dataset
import mnist
from rftree_builder import feature_importance as rf_feature_importance
from random_forest import RandomForestBuilder


def _random_forest_it(train_dataset, test_dataset, random_fores_builder, **kwargs):
    """
    Fits a RandomForest and compute its accuracy
    :param train_dataset: Dataset with the training samples
    :type train_dataset: Dataset
    :param test_dataset: Dataset with the testing samples
    :type test_dataset: Dataset
    :param random_fores_builder: RandomForestBuilder object
    :type random_fores_builder: RandomForestBuilder
    :param kwargs: RandomForestBuilder fit method keyword parameters
    :return:
    """
    ti = time()
    rf = random_fores_builder.fit(train_dataset, **kwargs)
    print(f'random forest done, elapsed time {time() - ti}s')

    predicted_labels = [pred[0] for pred in rf.predict(test_dataset.data)]
    num_correct_predictions = np.sum(predicted_labels == test_dataset.labels)
    accuracy = num_correct_predictions / float(len(test_dataset))
    print("Accuracy {} %".format(100 * np.round(accuracy, decimals=4)))


def test_iris():
    iris = sklearn.datasets.load_iris()
    dataset = Dataset(iris.data, iris.target)

    num_samples, num_features = dataset.shape
    # 150, 4
    max_depth = 10  # maxim nombre de nivells de l'arbre
    min_split_size = 5  # <5 mostres en un node i ja no el dividim
    sample_ratio = 0.7  # 0 < r <=1, cada arbre l'entrenem amb una fraccio de
    # totes les mostres d'entrenament agafades amb reposicio
    # = bagging
    num_trees = 10  # nombre d'arbres de decisio a fer
    num_features_node = int(np.sqrt(num_features))
    # nombre de features diferents a considerar en cada node,
    # heuristica que funciona be sovint

    rf = RandomForestBuilder(num_trees, sample_ratio=sample_ratio, max_depth=max_depth, min_split_size=min_split_size)

    ratio_train_test = 0.8
    # 80% train, 20% test

    idx = np.random.permutation(range(num_samples))
    # aixo fa 0,1,...,149 barrejats aleatoriament = "shuffle"
    # cal fer-ho perque les mostres carregades estan per classe

    num_samples_train = int(num_samples * ratio_train_test)
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train:]
    train_data, train_labels = dataset.data[idx_train], dataset.labels[idx_train]
    test_data, test_labels = dataset.data[idx_test], dataset.labels[idx_test]
    train_dataset = Dataset(train_data, train_labels)
    test_dataset = Dataset(test_data, test_labels)
    num_samples_test = len(test_labels)

    _random_forest_it(train_dataset, test_dataset, rf)
    return 0


def _test_mnist(get_mnist_fn, values):
    train, train_labels, test, test_labels = get_mnist_fn()
    train_dataset = Dataset(train, train_labels)
    test_dataset = Dataset(test, test_labels)

    num_trees = 10
    sample_ratio = 1
    # max_depth = 10
    # max_density = 10
    min_split_size = math.sqrt(len(train) * sample_ratio)
    # min_split_size = math.sqrt(len(train) * sample_ratio) / math.log(len(train) * sample_ratio)
    # min_split_size = math.log(len(train) * sample_ratio, 2)
    # num_features = int(math.sqrt(train_dataset.shape[1]) * 2)
    # min_info_gain = 5e-2

    criterion = Gini()
    feature_importance = rf_feature_importance(train_dataset, criterion, values=values)

    # def post_prune_predicate(split_node_dec):
    #     if split_node_dec.depth_level > 10:
    #         min_score = 0.1 * (split_node_dec.depth_level - 10)
    #         if split_node_dec.score < min_score:
    #             return True
    #     return False

    # def post_prune_predicate(split_node_dec):
    #     return len(split_node_dec) < min_split_size

    rfb = RandomForestBuilder(num_trees, sample_ratio=sample_ratio, min_split_size=min_split_size)
    for _ in range(1):
        _random_forest_it(train_dataset, test_dataset, rfb, values=values, feature_importance=feature_importance)

    return 0


def test_bare_mnist(values=range(32, 255, 32)):
    _test_mnist(mnist.bare, values)
    return 0


def test_projected_mnist(values=range(357, 7139, 357)):
    _test_mnist(mnist.projected, values)
    return 0


info_handler = logging.StreamHandler(stream=stdout)
info_handler.setLevel(logging.INFO)
error_handler = logging.StreamHandler(stream=stderr)
error_handler.setLevel(logging.WARN)
logging.getLogger('mnist').addHandler(info_handler)
logging.getLogger('mnist').addHandler(error_handler)
logging.getLogger('rftree_builder').addHandler(info_handler)
logging.getLogger('rftree_builder').addHandler(error_handler)
logging.getLogger('random_forest').addHandler(info_handler)
logging.getLogger('random_forest').addHandler(error_handler)

if __name__ == '__main__':
    test_bare_mnist()
