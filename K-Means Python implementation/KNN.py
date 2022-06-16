__authors__ = ['1565824', '1496672']
__group__ = 'Grup15'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        X = train_data.astype(np.float64, copy=False)
        self.train_data = X.reshape(X.shape[0], -1)


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        test_data = test_data.astype(np.float64, copy=False)
        test_data = test_data.reshape(test_data.shape[0], -1)
        distances = cdist(test_data, self.train_data, metric='euclidean')
        indices = distances[:, :].argsort()  # [np.argsort(distances[i]) for i in range(len(distances))]
        nei = np.array([self.labels[indices[i]] for i in range(distances.shape[0])])
        self.neighbors = nei[:, :k]


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        likely_label = []
        prob_label = []
        for ix, row in enumerate(self.neighbors):
            words = {}
            for word in row:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
            selected = max(words, key=words.get)
            likely_label.append(selected)
            prob_label.append(words[selected] / sum(words.values()))

        return np.array(likely_label), np.array(prob_label)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
