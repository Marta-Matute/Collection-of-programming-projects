from __future__ import annotations

from collections import Counter
from numbers import Number
from typing import Any, Dict, Tuple

from numpy import ndarray, random as nprandom


class Dataset:
    """
    A dataset is a set of tabulated data, where each row is a sample of the dataset and each column a feature of the
    samples. Each sample has a label, and generally there is one data structure for the samples and another for the
    labels.

    Class Dataset encapsulates a 2-d N x M ndarray of samples (N samples with M features) and its related 1-d N ndarray
    of labels (N labels) in a single object with convenient methods to perform operations over that structures.
    """

    def __init__(self, data: ndarray, labels: ndarray):
        """
        :param data: 2-d N x M ndarray of samples (N samples with M features)
        :param labels: 1-d N ndarray of labels (N labels)
        """

        self.data: ndarray = data
        self.labels: ndarray = labels

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    @property
    def shape(self):
        return self.data.shape

    def subset(self, ratio_samples: float, replacement=True) -> Dataset:
        """
        Creates a subset of this dataset object with random samples.

        :param ratio_samples: The ratio between the length of the subset and the dataset.
        :param replacement: Whether the samples taken from the dataset can be taken only once, or many times.
                            If True, each sample can be taken many times, otherwise only once.
                            Note: If set to false, the maximum value ratio_samples can take is 1.
                            Default = True

        :return: A Dataset created as a subset of this object
        """
        new_idx = nprandom.choice(range(len(self)), int(ratio_samples * len(self)), replacement)
        data_subset = self.data[new_idx]
        labels_subset = self.labels[new_idx]
        return Dataset(data_subset, labels_subset)

    def split(self, feature: int, value) -> Tuple[Dataset, Dataset]:
        """
        Creates two dataset object, one with the samples that its feature's value is lesser or equals than the value
        given as argument of the feature given as a argument, and the other with the samples that do not satisfy that
        condition.

        Is the value given as argument is not a number, it will test for equality instead of lesser or equals

        :param feature: feature to test for each sample
        :param value: value to test for the feature
        :return: A tuple with the two created dataset, the first one with the samples that satisfy the condition and the
                 second with the remaining samples
        """
        left, right = split(feature, value, self.data, self.data, self.labels)

        left_dataset = Dataset(left[0], left[1])
        right_dataset = Dataset(right[0], right[1])

        return left_dataset, right_dataset

    def split_labels(self, feature: int, value) -> Tuple[ndarray, ndarray]:
        """
        Creates two 1-d ndarray with the same length of this dataset, one with the labels of the samples that its
        feature's value is lesser or equals than the value given as argument of the feature given as a argument, and the
        other with the labels of the samples that do not satisfy that condition.

        Is the value given as argument is not a number, it will test for equality instead of lesser or equals

        :param feature: feature to test for each sample
        :param value: value to test for the feature
        :return: A tuple with the two created ndarray, the first one with the labels of the samples that satisfy the
        condition and the second with the remaining labels
        """
        left, right = split(feature, value, self.data, self.labels)

        return left[0], right[0]

    def label_counter(self) -> Dict[Any, int]:
        """
        Creates a dict with each different label as key an the number of times it appears as value

        :return: The created dict
        """
        return Counter(self.labels)

    def label_frequencies(self) -> Dict[Any, float]:
        """
        Creates a dict with each different label as key an its frequency (0, 1] within the labels.

        :return: The created dict
        """
        return label_frequencies(self.labels)

    def most_frequent_label(self):
        """
        :return: The most frequent label within the labels of this dataset
        """
        return most_frequent_label(self.labels)


def split(feature: int, value, data: ndarray, *ndarrays: ndarray) -> Tuple[Tuple[ndarray, ...], Tuple[ndarray, ...]]:
    """
    Creates two ndarray for each ndarray positional argument, one with the values of the ndarray positional argument
    with the same indices that the indices of the data argument samples that its feature's value is lesser or equals
    than the value given as argument of the feature given as a argument, and the other with remaining ndarray positional
    argument values.

    :param feature: feature to test for each sample of the data argument
    :param value: value to test for the feature
    :param data: ndarray with the samples
    :param ndarrays: narrays to split (must have same length as data)
    :return: Tuple with two elements, each with a tuple of ndarrays. The first one with a tuple of ndarrays with the
             values of the samples of each ndarray given as positional arguments that satisfy the condition and the
             second with the remaining values of those ndarrays.
    """
    if isinstance(value, Number):
        accepted = data[:, feature] <= value
        rejected = data[:, feature] > value
    else:
        accepted = data[:, feature] == value
        rejected = data[:, feature] != value

    left = tuple(map(lambda array: array[accepted], ndarrays))
    right = tuple(map(lambda array: array[rejected], ndarrays))

    return left, right


def label_counter(labels: ndarray) -> Dict[Any, int]:
    """
    Creates a Counter with each different label as key an the number of times it appears as value of the given labels.

    :param labels: A collection of labels from which make the dict
    :return: The created dict
    """
    return Counter(labels)


def label_frequencies(labels: ndarray) -> Dict[Any, float]:
    """
    Creates a dict with each different label as key an its frequency (0, 1] within the labels given as argument.

    :param labels: A collection of labels from which make the dict
    :return: The created dict
    """
    counter = Counter(labels)
    total = sum((count for label, count in counter.items()))
    frequencies = {label: (count / total) for label, count in counter.items()}
    return frequencies


def most_frequent_label(labels: ndarray):
    """
    :param labels: A collection of labels
    :return: The most frequent label within the labels given as argument
    """
    return max(label_frequencies(labels).items(), key=lambda item: item[1])
