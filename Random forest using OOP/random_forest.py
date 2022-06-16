import logging

from tqdm import tqdm

from logging_utils import get_handler_stream
from rftree_builder import RFAlgorithm, feature_importance as id3_feature_importance

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class RandomForest:
    """
    A random forest is a collection of decision trees built with a randomness component to give a more accurate
    prediction than a single decision tree for items not used to train the trees.

    See https://en.wikipedia.org/wiki/Random_forest

    Attributes
        trees: Sequence
               The collection of trees from which make predictions about items
    """
    def __init__(self, trees):
        self.trees = trees

    def predict(self, samples):
        """
        Predict a label for the given samples.

        :param samples: Sequence of samples. Each sample is a Sequence of values of each feature of the sample
        :return: List with the prediction of each sample. Each prediction is a tuple with the predicted label and its
        frequency in the leaf node the sample reached.
        """
        predictions = []
        for sample in samples:
            sample_predictions = [tree.predict(sample) for tree in self.trees]
            majoritary_prediction = max(sample_predictions, key=lambda prediction: prediction[1])
            predictions.append(majoritary_prediction)
        return predictions


class RandomForestBuilder:
    """
    Random forest building algorithm

    Attributes:
        num_trees: int
            The number of trees of the random forests to build

        sample_ratio: float
            The ratio of samples of the dataset to use to build a random forest

        tree_builder: DecisionTreeBuilder
            The DecisionTreeBuilder object which from to build decision trees
    """

    def __init__(self, num_trees, **kwargs):
        """
        The num_trees parameter and the sample_ratio keyword parameter are for the RandomForestBuilder instantiation,
        the remaining keyword parameters are for the DecisionTreeBuilder instantiation in case that there is not
        provided a DecisionTreeBuilder object through the tree_builder keyword parameter.

        :param num_trees: The number of trees to make for building random forests
        :type num_trees: int

        :key sample_ratio: float -
            If provided, the ratio of samples of the dataset to use to build a random forest. Default = 0.7

        :key criterion: Union[Criterion, type, str] -
            If provided, a Criterion object, type or class name from which computing scores and information gains.
            Default = 'Gini'

        :key tree_builder: Union[DecisionTreeBuilder, type, str] -
            If provided, a DecisionTreeBuilder object, type, or class name with from which build the random forest
            trees.
            Default = 'RFAlgorithm'

        :key max_depth: int -
            If provided, max depth allowed for built trees (pre-pruning)

        :key min_split_size: int -
            If provided, min number of samples allowed for split in a node to be able to become a split node. If a node
            number of samples is less than min_split_size, it will become a leaf (pre-pruning). Default = 1

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
            Default =  int(sqrt(len(dataset)))

        :key post_prune_predicate: Callable[[SplitNodeDecorator], bool] -
            If provided, a function that accepts a SplitNodeDecorator and returns True if it has to be pruned, False
            otherwise. (post-pruning)
        """

        self.num_trees = num_trees
        self.sample_ratio = kwargs.get('sample_ratio', 0.7)

        tree_builder = kwargs.get('tree_builder', 'RFAlgorithm')
        if isinstance(tree_builder, str):
            if tree_builder == 'RFAlgorithm':
                tree_builder = RFAlgorithm
            else:
                raise NotImplementedError('Not implemented tree builder')

        if isinstance(tree_builder, type):
            self.tree_builder = tree_builder(**kwargs)

        _logger.debug(f'{self.__class__.__name__} created. Attributes:')
        for name, value in self.__dict__.items():
            if not name.startswith('_') and value is not None:
                _logger.debug(f'    {name} = {value}')

    def fit(self, dataset, **kwargs):
        """
        Builds a random forest from the given dataset

        :param dataset: Dataset from which to build the random forest
        :type dataset: Dataset

        :key feature_importance: Union[ndarray, str] -
            If provided, A 1d ndarray with the probabilities of each feature to be chosen, or the 'compute' str to make
            the RandomForestBuilder to compute the feature importance using the feature_importance function of the id3
            module.
            The feature_importance ndarray can be provided using the id3 module feature_importance function.

        :key values: Collection -
            If provided, a collection of values to use for splitting the dataset when trying to find the best split.
            Very highly recommended for a large dataset.

        :key undecore: bool -
            The ID3 algorithm will use decorated versions of Node classes for building the tree.
            If undecore is True, then the build method will return a tree with the non-decorated Node classes, otherwise
            it will return the tree with the decorated versions of Node classes. Default = True

        :return: RandomForest
        """
        trees = []

        feature_importance = kwargs.get('feature_importance', None)
        if feature_importance is not None and isinstance(feature_importance, str) and feature_importance == 'compute':
            kwargs['feature_importance'] = id3_feature_importance(dataset, self.tree_builder.criterion)

        _logger.info(f'Building a random forest with {self.num_trees} trees')

        file = get_handler_stream(_logger, logging.INFO)
        for _ in tqdm(range(self.num_trees), file=file) if file is not None else range(self.num_trees):
            subset = dataset.subset(self.sample_ratio)
            tree = self.tree_builder.build(subset, **kwargs)
            trees.append(tree)

        return RandomForest(tuple(trees))
