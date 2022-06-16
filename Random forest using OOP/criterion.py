from abc import ABC, abstractmethod

from numpy import ndarray

from dataset import label_frequencies


class Criterion(ABC):
    """
    Abstract class representing an object with a score and a gain method.

    Both score and gain methods are used as metrics to know how much mixed up are the labels in a dataset (score), and
    how much information is gained after splitting a dataset (gain).

    Classes implementing Criterion must follow the following conventions:
        The score method must return a lower value when the labels are less mixed up, and a higher value when the labels
        are more mixed up.

        The gain method must return a higher value when the information gain is higher, and a lower value when the
        information gain is lower.

        Both should return a float between 0 and 1, both included.
    """

    @abstractmethod
    def score(self, label_freq):
        pass

    @abstractmethod
    def gain(self, parent_score, left_label_freq, right_label_freq):
        pass


class Gini(Criterion):
    """
    A Criterion implementation.

    Extract from Wikipedia:

    Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it
    was randomly labeled according to the distribution of labels in the subset.



    Link: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

    The gini impurity can be obtained through the score method, the information gain through the gain method.
    """

    def score(self, labels: ndarray) -> float:
        # 1 - sum(label_frequency^2 for label_frequency in label_frequencies)
        return 1 - sum(map(lambda lf: lf[1]**2, label_frequencies(labels).items()))

    def gain(self, parent_score: float, left_labels: ndarray, right_labels: ndarray) -> float:
        n_samples = len(left_labels) + len(right_labels)
        left_score = self.score(left_labels)
        right_score = self.score(right_labels)
        gain = parent_score - (len(left_labels) * left_score + len(right_labels) * right_score) / n_samples
        return gain
