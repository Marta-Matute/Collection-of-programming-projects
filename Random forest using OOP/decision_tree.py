from __future__ import annotations  # Necessari per a que Node pugui veure SplitNode

from abc import ABC, abstractmethod
from typing import Any, Tuple, Sequence


class DecisionTree:
    """
    Represents a decision tree structure.

    A decision tree as a predictive model is a prediction tool that uses a tree-like model to go from observations about
    an item (represented in the branches) to conclusions about the item (represented in the leaves).


    See: https://en.wikipedia.org/wiki/Decision_tree
         https://en.wikipedia.org/wiki/Decision_tree_learning

    Properties
        root: Node
            The root node of the decision tree
    """

    def __init__(self, root: Node):
        self._root = root

    @property
    def root(self):
        return self._root

    def predict(self, sample: Sequence) -> Tuple[Any, float]:
        """
        Predict a label for a sample.
        :param sample: sequence of values of each feature of the sample
        :return: tuple with the predicted label and its frequency in the leaf node the sample reached.
        """
        if self.root is None:
            raise AssertionError('The DecisionTree has not been built yet.')
        return self.root.predict(sample)


class Node(ABC):
    """
    Abstract class that represents a decision tree node.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, sample: Sequence) -> Tuple[Any, float]:
        """
        Predict a label for a sample.

        :param sample: sequence of values of each feature of the sample
        :return: tuple with the predicted label and its frequency in the leaf node the sample reached.
        """
        pass


class SplitNode(Node):
    """
    Represents a node where the dataset gets split according to a feature and a feature's value.

    Attributes
        feature: int
            The feature used to test a dataset items

        value: The feature value used to test a dataset items

        left: Node
            Branch to the Node to follow when an item passes the test for this node feature and value

        right: Node
            Branch to the Node to follow when an item doesn't pass the test for this node feature and value
    """

    def __init__(self, feature: int, value, left, right):
        super().__init__()
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

    def predict(self, sample: Sequence) -> Tuple[Any, float]:
        """
        Predict a label for a sample.

        :param sample: sequence of values of each feature of the sample
        :return: tuple with the predicted label and its frequency in the leaf node the sample reached.
        """
        if sample[self.feature] < self.value:
            return self.left.predict(sample)
        else:
            return self.right.predict(sample)


class LeafNode(Node):
    """
    Represents a leaf node of a decision Tree. A leaf node is a node that will not get split.

    Attributes
        label_frequencies: dict {label: frequency}
            A dictionary with labels as keys, and its frequency from 0 (excluded) to 1 (included)

        most_frequent_label: The most frequent label of the leaf
    """

    def __init__(self, label_frequencies):
        super().__init__()
        self.label_frequencies = label_frequencies
        self.most_frequent_label = max(self.label_frequencies.items(), key=lambda freq: freq[1])

    def predict(self, sample: Sequence) -> Tuple[Any, float]:
        """
        Predict a label for a sample.

        :param sample: sequence of values of each feature of the sample
        :return: tuple with the predicted label and its frequency in the leaf node the sample reached.
        """
        return self.most_frequent_label  # (label, frequency)


class DecisionTreeBuilder(ABC):
    """
    Abstract class representing a decision tree building algorithm
    """

    @abstractmethod
    def build(self, dataset) -> DecisionTree:
        pass
