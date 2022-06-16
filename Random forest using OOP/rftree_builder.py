from __future__ import annotations
from abc import ABC
from collections import deque
from enum import auto, Enum
import logging
from math import sqrt
from typing import Collection, Sequence, Tuple, Any

from numpy import ndarray, random as nprandom, empty as npempty
from sortedcontainers import SortedDict
from tqdm import tqdm

from criterion import Criterion, Gini
from dataset import Dataset, label_frequencies
from decision_tree import Node, SplitNode, LeafNode, DecisionTree, DecisionTreeBuilder
from logging_utils import get_handler_stream

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class NodeDecorator(Node, ABC):
    """
    Decorates a Node with metrics information for pruning purpose

    Attributes
        decorated: Node

        score: float
            Number in the interval [0, 1] indicating how much shuffled are the labels at this node. The lower the score,
            the better score

        parent: Optional[SplitNodeDecorator]
    """

    class Side(Enum):
        LEFT = auto()
        RIGHT = auto()

    def __init__(self, decorated: Node, score: float, parent: SplitNodeDecorator = None, side=Side.LEFT):
        """
        :param decorated: The Node to decorate
        :param score: Computed score for the dataset represented by this node
        :param parent: The parent of the created node, if it has one
        :param side: If this node has a parent, the side of the parent this node belongs to
        """
        super().__init__()
        self.decorated = decorated
        self.score = score
        self.parent = parent
        if parent is not None:
            if side == NodeDecorator.Side.LEFT:
                parent.left = self  # this NodeDecorator to the left side of its SplitNodeDecorator parent
                parent.decorated.left = decorated  # this decorated Node to the left side of its SplitNode Parent
            else:  # Right
                parent.right = self  # this NodeDecorator to the right side of its SplitNodeDecorator parent
                parent.decorated.right = decorated  # this decorated Node to the right side of its SplitNode Parent

    def predict(self, sample: Sequence) -> Tuple[Any, float]:
        """
        Predict a label for a sample.
        :param sample: sequence of values of each feature of the sample
        :return: tuple with the predicted label and its frequency in the leaf node the sample reached.
        """
        return self.decorated.predict(sample)


class SplitNodeDecorator(NodeDecorator):
    """
    Decorates a SplitNode with metrics information for pruning purpose

    Attributes
        NodeDecorator abstract class inherited attributes

        feature: int
            The feature used to split the dataset

        value: Any
            The feature value used to split the dataset

        left_dataset: Dataset
            The left split of a Dataset

        right_dataset: Dataset
            The right split of a Dataset

        depth_level: int
            The depth where this node abides. (The root node is at depth level 0)

        child_depth: int
            How deep this node child reach

        length: int
            length of the dataset it represents

        weight_parent: float
            The weight of this node with respect to its parent.
            Ratio between its length and its parent length (self.length / parent.length)

        weight_tree: float
            The weight of this node with respect to the whole tree.
            Ratio between its length and the root node (or self.weight_parent * parent.weight_tree)

        gain: float
            Number in the interval [0, 1] indicating how much information is gained splitting the dataset
            with this node feature and value. The higher the value, higher the information gain is

        left: NodeDecorator
            Once the tree is built, represents the left child node of this node. While building the tree, could
            represent any data the algorithm may need to do its work

        right: NodeDecorator
            Once the tree is built, represents the right child node of this node. While building the tree, could
            represent any data the algorithm may need to do its work.
    """

    def __init__(self, decorated: SplitNode, left_dataset: Dataset, right_dataset: Dataset, length: int, score: float,
                 gain: float, parent: SplitNodeDecorator = None, side=NodeDecorator.Side.LEFT):
        """
        :param decorated: The split node to decorate
        :param left_dataset: Left split dataset of the split node
        :param right_dataset: Right split dataset of the split node
        :param length: Length of the dataset it represents
        :param score: Computed score for the dataset represented by this node
        :param gain: Number in the interval [0, 1] indicating how much information is gained splitting the dataset
            with this node feature and value. The higher the value, higher the information gain is
        :param parent: The parent of the created node, if it has one
        """
        super().__init__(decorated, score, parent, side)
        self.left, self.right = None, None
        self.left_dataset, self.right_dataset = left_dataset, right_dataset

        self.length = length
        if self.parent is not None:
            self.depth_level = self.parent.depth_level + 1
            self.weight_parent = self.length / self.parent.length
            self.weight_tree = self.weight_parent * self.parent.weight_tree
        else:
            self.depth_level = 0
            self.weight_parent = 1
            self.weight_tree = 1

        self._increase_ancestors_child_count()

        self.child_depth = 1
        self._increase_ancestors_child_depth()
        self.child_count = 0

        self.gain = gain

    def _increase_ancestors_child_count(self):
        parent = self.parent
        while parent is not None:
            parent.child_count += 1
            parent = parent.parent

    def _increase_ancestors_child_depth(self):
        # Increments by 1 the child_depth attribute of this node ancestors
        actual, parent = self, self.parent
        while parent is not None:
            parent.child_depth = actual.child_depth + 1
            actual = parent
            parent = parent.parent

    def __len__(self):
        return self.length


class LeafDecorator(NodeDecorator):
    """
    Decorates a Leaf with the labels it represents for pruning purpose

    Attributes
        NodeDecorator abstract class inherited attributes

        labels: ndarray of the labels of the leaf
    """

    def __init__(self, decorated: LeafNode, labels: ndarray, score: int, parent: SplitNodeDecorator = None,
                 side=NodeDecorator.Side.LEFT):
        """
        :param decorated: The split node to decorate
        :param labels: The labels of the leaf
        :param score: Computed score for the dataset represented by this node
        :param parent: The parent of the created node, if it has one
        """
        super().__init__(decorated, score, parent, side)
        self.labels = labels


class RFAlgorithm(DecisionTreeBuilder):
    """
    Random forest decision tree building algorithm

    Attributes
        criterion: Criterion
            A Criterion object for computing scores and information gain. By default uses a Gini object.

        max_depth: int
            If provided, max depth allowed for built trees (pre-pruning)

        min_split_size: int
            If provided, min number of samples allowed for split in a node to be able to become a split node. If a node
            number of samples is less than min_split_size, it will become a leaf (pre-pruning)

        min_score: float
            If provided, number in the interval [0, 1] representing the minimum score possible for a split node.
            If a node score is less than min_score, the node will become a leaf (pre-pruning)

        min_info_gain: float
            If provided, number in the interval [0, 1] representing the minimum information gain possible for a split
            node. A candidate to split node that would have a gain of less than min_info_gain will become a leaf
            (pre-pruning)

        max_nodes: int
            If provided, maximum number of nodes a built tree can have. (pre-pruning)

        num_features: int
            If provided, number of features to use to find the best split possible.
            If not given it will use int(sqrt(len(dataset))) as default

        post_prune_predicate: function(SplitNodeDecorator) -> bool
           If provided, a function that accepts a SplitNodeDecorator and returns True if it has to be pruned, False
           otherwise. (post-pruning)

        values: Collection
            If provided, a collection of values to use for splitting the dataset when trying to find the best split.
            Very highly recommended for a large dataset.
    """

    def __init__(self, **kwargs):
        """
        :key criterion: Union[Criterion, str] -
            If provided, a Criterion object, type or class name from which computing scores and information gains.
            Default = 'Gini'

        :key max_depth: int -
            If provided, max depth allowed for built trees (pre-pruning)

        :key min_split_size: int -
            If provided, min number of samples allowed for split in a node to be able to become a split node. If a node
            number of samples is less than min_split_size, it will become a leaf (pre-pruning).
            Default = 1

        :key min_score: float -
            If provided, number in the interval [0, 1] representing the minimum score possible for a split node.
            If a node score is less than min_score, the node will become a leaf (pre-pruning)

        :key min_info_gain: float -
            If provided, number in the interval [0, 1] representing the minimum information gain possible for a split
            node. A candidate to split node that would have a gain of less than min_info_gain will become a leaf
            (pre-pruning)

        :key: max_nodes: int -
            If provided, maximum number of nodes a built tree can have. (pre-pruning)

        :key num_features: int -
            If provided, number of features to use to find the best split possible.
            Default = int(sqrt(len(dataset)))

        :key post_prune_predicate: Callable[[SplitNodeDecorator], bool] -
            If provided, a function that accepts a SplitNodeDecorator and returns True if it has to be pruned, False
            otherwise. (post-pruning)
        """

        criterion = kwargs.get('criterion', 'Gini')
        if isinstance(criterion, str):
            if criterion == 'Gini':
                criterion = Gini
            else:
                raise NotImplementedError('Not implemented criterion')

        if isinstance(criterion, type):
            self.criterion = criterion()

        self.max_depth = kwargs.get('max_depth', None)
        self.min_split_size = kwargs.get('min_split_size', 1)
        self.min_score = kwargs.get('min_score', 1e-12)
        self.min_info_gain = kwargs.get('min_info_gain', None)

        max_density = kwargs.get('max_density', None)
        self.max_nodes = (2**max_density - 1) if max_density is not None else kwargs.get('max_nodes', None)

        self.num_features = kwargs.get('num_features', lambda num: int(sqrt(num)))

        self.post_prune_predicate = kwargs.get('post_prune_predicate', None)

        self._values = None
        self._feature_importance = None

        if not isinstance(self.num_features, int):
            try:
                n = self.num_features(10)
                if not isinstance(n, int):
                    raise ValueError
            except Exception as _:
                raise ValueError('num_features kwargs must be a int or a function with signature int -> int')

        _logger.debug(f'{self.__class__.__name__} created. Attributes:')
        for name, value in self.__dict__.items():
            if not name.startswith('_') and value is not None:
                _logger.debug(f'    {name} = {value}')

    def build(self, dataset: Dataset, **kwargs):
        """
        Builds a decision tree from the given dataset

        :param dataset: Dataset from which to build the decision tree

        :key feature_importance: ndarray - If provided, A 1d ndarray with the probabilities of each feature to be
            chosen. The feature_importance ndarray can be provided using the ID3 feature_importance method

        :key values: Collection - If provided, a collection of values to use for splitting the dataset when trying to
            find the best split. Very highly recommended for a large dataset.

        :key undecore: bool - The ID3 algorithm will use decorated versions of Node classes for building the tree.
            If undecore is True, then the build method will return a tree with the non-decorated Node classes, otherwise
            it will return the tree with the decorated versions of Node classes. Default = True

        :return: DecisionTree
        """

        self._values = kwargs.get('values', None)
        if self._values is not None:
            self._values = set(self._values)

        num_features_backup = self.num_features
        if not isinstance(self.num_features, int):
            self.num_features = self.num_features(dataset.shape[1])

        self._feature_importance = kwargs.get('feature_importance', None)

        score = self.criterion.score(dataset.labels)
        root = self._make_node(1, dataset, score)

        # See _build comments
        sd = SortedDict({score: [root]} if isinstance(root, SplitNodeDecorator) else {})

        self._build(sd)
        self._feature_importance = None
        self._values = None

        if self.post_prune_predicate is not None:
            _post_prune(root, self.post_prune_predicate)

        self.num_features = num_features_backup

        if kwargs.get('undecore', True):
            root = root.decorated

        return DecisionTree(root)

    def _build(self, sd):
        # It will be used a SortedDict with scores as keys and lists of SplitNodeDecorator as values to build the tree
        # like using a "queue" (breadth-first). But instead of computing first the first in, it will compute first the
        # SplitNodeDecorator with worst score.

        def add(score, node):
            if isinstance(node, SplitNodeDecorator):
                if score not in sd:
                    sd[score] = [node]
                else:
                    sd[score].append(node)

        def pop():
            score, nodes = sd.popitem()
            node = nodes[0]
            if len(nodes) > 1:
                sd[score] = nodes[1:]
            return node

        def build():
            num_node = 1
            while len(sd) > 0:
                # decorator: decorator node, decorated: decorated node
                decorator = pop()  # pops the SplitNodeDecorator with worst score

                left_dataset = decorator.left_dataset
                decorator.left_dataset = None  # to not store a split of the dataset that is not needed anymore

                right_dataset = decorator.right_dataset
                decorator.right_dataset = None  # to not store a split of the dataset that is not needed anymore

                left_score = self.criterion.score(left_dataset.labels)
                right_score = self.criterion.score(right_dataset.labels)

                num_node += 1
                left_node = self._make_node(num_node, left_dataset, left_score, decorator)

                num_node += 1
                right_node = self._make_node(num_node, right_dataset, right_score, decorator, NodeDecorator.Side.RIGHT)

                add(left_score, left_node)
                add(right_score, right_node)

        build()

    def _best_split(self, dataset, score):
        features = nprandom.choice(range(dataset.shape[1]), self.num_features, False, self._feature_importance)

        best_feature, best_value, best_gain = None, None, 0

        transposed = dataset.data.transpose() if self._values is None else None
        for feature in features:
            values = self._values if self._values is not None else set(transposed[feature])
            for value in values:
                left_labels, right_labels = dataset.split_labels(feature, value)
                gain = self.criterion.gain(score, left_labels, right_labels)
                if gain > best_gain:
                    best_feature, best_value, best_gain = feature, value, gain

        return best_feature, best_value, best_gain

    def _make_node(self, node_num, dataset, score, parent: SplitNodeDecorator = None, side=NodeDecorator.Side.LEFT):
        make_leaf = score < self.min_score
        make_leaf = make_leaf or len(dataset) < self.min_split_size
        make_leaf = (make_leaf or self.max_depth is not None and parent is not None
                     and parent.depth_level + 1 >= self.max_depth)
        make_leaf = make_leaf or self.max_nodes is not None and node_num >= self.max_nodes

        if not make_leaf:
            feature, value, gain = self._best_split(dataset, score)
            split = dataset.split(feature, value) if gain > 0 else None
            make_leaf = split is None
            if not make_leaf:
                left, right = split
                make_leaf = self.min_info_gain is not None and gain < self.min_info_gain
                make_leaf = make_leaf or len(left) == 0 or len(right) == 0
                if not make_leaf:
                    split_node = SplitNode(feature, value, None, None)
                    return SplitNodeDecorator(split_node, left, right, len(dataset), score, gain, parent, side)

        leaf = LeafNode(dataset.label_frequencies())
        return LeafDecorator(leaf, dataset.labels, score, parent, side)


def feature_importance(dataset: Dataset, criterion: Criterion, values: Collection = None) -> ndarray:
    """
    Compute the importance of each feature of a given dataset.

    The importance of a feature is how likely is to gain information when splitting the given database with that
    feature.

    This function returns a 1-d ndarray of the same length that the given database. Each position of the returned
    ndarray correspond to a feature (pos 0 -> 1st feature, pos 1 -> 2nd feature, etc). Each position of the returned
    ndarray contains a number in the range [0, 1], and the sum of all the values of the returned ndarray is
    (very approximately) equals to 1.

    That returned ndarray can be used as argument to the parameter p of the method numpy.random.choice. This will
    result in that the chosen features will be more likely to be relevant that the non-chosen ones.

    :param dataset: Dataset from which compute the feature importance
    :param criterion: A Criterion object from which compute the feature importance
    :param values: If given, A Collection of values to use for each feature to compute the feature importance. Highly
           recommended for large dataset.
    :return: ndarray. See this function description
    """
    samples, features = dataset.shape
    transposed = dataset.data.transpose() if values is None else None
    score = criterion.score(dataset.labels)
    mean_gain_sum = 0
    feature_mean_gain = npempty(features)
    _logger.info(f'Computing the feature importance of {features} features')
    file = get_handler_stream(_logger, logging.INFO)
    for feature in tqdm(range(features), file=file) if file is not None else range(features):
        mean_gain = 0
        values = values if values is not None else set(transposed[feature])
        for value in values:
            left_labels, right_labels = dataset.split_labels(feature, value)
            mean_gain += criterion.gain(score, left_labels, right_labels)
        mean_gain /= len(values)
        feature_mean_gain[feature] = mean_gain
        mean_gain_sum += mean_gain
    return feature_mean_gain / mean_gain_sum


def _post_prune(root, predicate):
    def assign_left():
        leaf = _prune_split_node(node)
        parent.left = leaf
        parent.decorated.left = leaf.decorated

    def assign_right():
        leaf = _prune_split_node(node)
        parent.right = leaf
        parent.decorated.right = leaf.decorated

    if isinstance(root, SplitNodeDecorator):
        if predicate(root):
            root = _prune_split_node(root)
        else:
            l_queue = deque()
            r_queue = deque()
            l_queue.append(root.left)
            r_queue.append(root.right)
            while len(l_queue) > 0 or len(r_queue) > 0:
                qf = (l_queue, assign_left), (r_queue, assign_right)
                for queue, assign in qf:
                    while len(queue) > 0:
                        node = queue.popleft()
                        parent = node.parent
                        if isinstance(node, SplitNodeDecorator):
                            if predicate(node):
                                assign()
                            else:
                                l_queue.append(node.left)
                                r_queue.append(node.right)


def _prune_split_node(split_node_decorator):
    labels = npempty(len(split_node_decorator))
    ini, fin = 0, None
    queue = deque()
    queue.append(split_node_decorator)
    while len(queue) > 0:
        node = queue.popleft()
        if isinstance(node, SplitNodeDecorator):
            queue.append(node.left)
            queue.append(node.right)
        else:
            fin = ini + len(node.labels)
            labels[ini:fin] = node.labels
            ini = fin

    leaf = LeafNode(label_frequencies(labels))
    return LeafDecorator(leaf, labels, split_node_decorator.score, split_node_decorator.parent)
