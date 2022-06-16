__authors__ = ['1565824', '1496672']
__group__ = 'Grup15'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """

        self._init_options(options)  # DICT options
        self.num_iter = 0
        self.K = K
        self._init_X(X)

    #############################################################
    #  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        # print("matrix shape = ", X.shape)
        if self.options['crop'] == 'cropped':
            center_matrix = X[20:60, 15:45]
            X = center_matrix
        X = X.astype(np.float64, copy=False)
        X = X.reshape(-1, X.shape[-1])
        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  # within class distance.
        if not 'crop' in options:
            options['crop'] = False

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options


    def _init_centroids(self):
        """
        Initialization of centroids
        """

        if self.options['km_init'].lower() == 'first':
            _, indices = np.unique(self.X, return_index=True, axis=0)  # _ = self.old_centroids?
            self.centroids = self.X[sorted(indices)[:self.K]]
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])

        elif self.options['km_init'].lower() == 'random':
            _, indices = np.unique(self.X, return_index=True, axis=0)
            self.centroids = self.X[np.random.choice(indices, self.K)]
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])

        elif self.options['km_init'].lower() == 'custom':
            # Diagonal elements of hypercube
            # self.X.diagonal() ??
            return

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """

        distances = distance(self.X, self.centroids)

        self.labels = distances.argmin(axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the
        points assigned to the centroid
        """
        self.old_centroids = self.centroids
        number_of_centroids = self.centroids.shape[0]

        self.centroids = np.array([self.X[self.labels == i, :].mean(axis=0) for i in range(number_of_centroids)])

    def converges(self):
        """
        Checks if there is a difference between current and old centroids

        """
        return np.allclose(self.centroids, self.old_centroids)

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        ITERATIONS = 0
        MAX_ITERATIONS = 100000
        self._init_centroids()

        while (not self.converges()):
            self.get_labels()
            self.get_centroids()
            ITERATIONS += 1
            if (ITERATIONS >= MAX_ITERATIONS):
                break

    def whitinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        return np.sum(np.linalg.norm(self.X - self.centroids[self.labels], axis=1)) / self.X.shape[0]

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        class_distances = []
        for k in range(1, max_K):
            self.K = k
            self.fit()
            class_distances.append(self.whitinClassDistance())
            if k > 1:
                dec = 100 * class_distances[k - 1] / class_distances[k - 2]
                if 100 - dec < 20:
                    self.K = k - 1
                    break


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    K = X.shape[0]
    D = C.shape[0]

    X_dist = (X * X).sum(axis=1).reshape((K, 1)) * np.ones(shape=(1, D))  # np.matmul(X, X)
    C_dist = (C * C).sum(axis=1) * np.ones(shape=(K, 1))
    dist = X_dist + C_dist - 2 * X.dot(C.T)

    # zero_control = np.less(dist, 0.0)
    # dist[zero_control] = 0.0
    return np.sqrt(dist)


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    probs = np.array(utils.get_color_prob(centroids))
    maxs = [np.argmax(probs[i], axis=0) for i in range(len(probs))]
    return [utils.colors[x] for x in maxs]


def get_colors_and_probs(km):
    """
    for each row of the numpy matrix 'centroids' returns the color label folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    (unique, counts) = np.unique(km.labels, return_counts=True)

    probs = np.array(utils.get_color_prob(km.centroids))
    maxs = [np.argmax(probs[i], axis=0) for i in range(len(probs))]
    color_list = np.array([utils.colors[x] for x in maxs])
    prob_list = np.max(probs, axis=1)

    color_dict = {}
    for i in range(len(color_list)):
        color = color_list[i]
        if color not in color_dict.keys():
            color_dict[color] = [prob_list[i], counts[i]]
        else:
            average_prob = np.array([color_dict[color][0], prob_list[i]]).mean(axis=0)
            total_counts = color_dict[color][1] + counts[i]
            color_dict[color] = [average_prob, total_counts]

    unique_colors_in_pred = np.array([x for x in color_dict.keys()])
    unique_prob_list = np.array([color_dict[x][0] for x in unique_colors_in_pred])
    unique_count_list = np.array([color_dict[x][1] for x in unique_colors_in_pred])

    labels_count_sorted = np.argsort(unique_count_list)[::-1]
    sorted_color_list = unique_colors_in_pred[labels_count_sorted]
    sorted_prob_list = unique_prob_list[labels_count_sorted]

    sorted_color_list = list(sorted_color_list)
    sorted_prob_list = list(sorted_prob_list)
    return sorted_color_list, sorted_prob_list
