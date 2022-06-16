import logging
from pickle import Unpickler
from zipfile import ZipFile

import numpy as np
from tqdm import tqdm

from logging_utils import get_handler_stream

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def _append(item, iterable):
    yield from iterable
    yield item


def _flat(nested):
    return (item for iterable in nested for item in iterable)


def bare():
    """
    Loads the mnist.pkl.zip file and reads the dataset.

    :param args: List with 0, 1 or 2 optional arguments. The first item, if given must be a float in range [0, 1] with
                 the ratio of the items to load from the MNIST dataset. The second item, if given, must be a boolean
                 value. True to choose random items from the MNIST dataset, False to read items sequentially from the
                 MNIST dataset.

    :return: Tuple with 4 items. The training data, the training labels, the testing data and the testing labels.
    """
    with ZipFile('mnist.pkl.zip') as mnist_zip:
        with mnist_zip.open('mnist.pkl') as mnist_pkl:
            mnist_unpkl = Unpickler(mnist_pkl)
            mnist_dict = mnist_unpkl.load()

            train = mnist_dict['training_images']
            train_labels = mnist_dict['training_labels']
            test = mnist_dict['test_images']
            test_labels = mnist_dict['test_labels']
            return train, train_labels, test, test_labels


def projected():
    """
    Loads the mnist.pkl.zip file and reads the dataset.

    From each item with 784 features (28x28 pixels, 28 rows and 28 columns) will make a projection of each row (28) and
    a projection of each column (28). That is, each item will lessen its features from 784 to 56.

    The projection of a row is the sum of the values of the row, and the projection of a column is the sum of the values
    of the column.
    Shoud be noted that the values a feature of the original item can be between 0 and 255, and the values of a feature
    of a projected item can be between 0 and 28 * 255 = 7140.

    :return: Tuple with 4 items. The training projected data, the training labels, the testing projected data and the
             testing labels.
    """
    with ZipFile('mnist.pkl.zip') as mnist_zip:
        with mnist_zip.open('mnist.pkl') as mnist_pkl:
            mnist_unpkl = Unpickler(mnist_pkl)
            mnist_dict = mnist_unpkl.load()

            def from_ndarray(images):
                def sum_row_pixels(_row):
                    return (np.sum(_28) for _28 in (_row[_i:(_i + 28)] for _i in range(0, 28 * 28, 28)))

                def sum_col_pixels(_row):
                    def take_28(col: int):
                        return _row[np.array(range(col, 28 * 28, 28))]

                    return (np.sum(_28) for _28 in (take_28(_i) for _i in range(28)))

                rows = images.shape[0]
                data = np.empty((rows, 56))
                _logger.info(f'Projecting {len(images)} items.')
                file = get_handler_stream(_logger, logging.INFO)
                for i in tqdm(range(len(images)), file=file) if file is not None else range(len(images)):
                    array_row = images[i]
                    _row = np.array([item for item in _flat((sum_row_pixels(array_row), sum_col_pixels(array_row)))])
                    data[i] = _row

                return data

            train, train_labels = from_ndarray(mnist_dict['training_images']), mnist_dict['training_labels']
            test, test_labels = from_ndarray(mnist_dict['test_images']), mnist_dict['test_labels']
            return train, train_labels, test, test_labels
