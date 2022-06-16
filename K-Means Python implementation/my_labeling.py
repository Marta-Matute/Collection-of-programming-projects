__authors__ = ['1565824', '1496672']
__group__ = 'Grup15'

import numpy as np
import copy

def filter_colors(kmeans, color_list, color_scores=None):
    """
    :param kmeans: kmeans object
    :param color_list: list of color names obtainted form the kmeans object
    :return: the filtered color_list (removed duplicates, removed irrelevant ....???)
             and the score of the colors (as color_naming returns)
    """
    if type(color_list) is not list:
        color_list = [color_list]
    if color_scores is None:
        color_scores = np.ones(len(color_list))

    colors = copy.deepcopy(color_list)

    #####################################################
    #   Keep the remaining code intact
    ######################################################
    scores = [np.array([s for c, s in zip(color_list, color_scores) if c == col]).mean() for col in colors]

    return colors, np.array(scores)


def get_accuracy(predicted_class_labels, test_labels):
    """
    :param predicted_class_labels: Nx1 vector with predicted class per sample
    :param test_labels:  Nx1 vector with real class per sample
    :return: accuracy performance (ie % of corrected labelled items)
    """
    return len(predicted_class_labels[predicted_class_labels == test_labels]) / len(predicted_class_labels)


def precision_at_k(query, real_labels):
    """
    :param query: (query_class, query_color) class string, list of color names
    :param real_labels: (real_class_labels, real_color_labels)
    :return: precision of the query at k first instances--> % of correct classified samples
    
    
    filter_1 = [(class_label, color_label) for class_label, color_label in
                zip(real_labels[0], real_labels[1]) if class_label == query[0]]  # from query those that match the class
    l = len([color_label for color_label in filter_1 if color_label == query[1]])  # from filter1 those that match color
    k_precision = l / len(query)
    
    return k_precision"""
    q0,q1 = query
    rl0, rl1 = real_labels
    
    q0 = [q0]
    if type(q1) is not list:
        q1 = list(q1)
    if type(rl0) is not list:
        rl0 = list(rl0)
    if type(rl1) is not list:
        rl1 = list(rl1)
    
    l1 = [1 if len(set(q0).intersection(set(rl0)))> 0 else 0 for _ in range(len(rl0))]
    l2 = [1 if len(set(q1).intersection(set(rl1[i])))> 0 else 0 for i in range(len(rl1))]
    #l2 = ([len(set(q1).intersection(set(rl))) for rl in rl1])
    total = sum(l1) + sum(l2)
    if total:
        return total/(len(rl0) + len(rl1))
    return 0


def print_acc_by_class(predicted_class_labels, test_labels):  #
    """
    :param predicted_class_labels:
    :param test_labels:
    :return: None
    """
    cl = np.array(predicted_class_labels)
    tl = np.array(test_labels)
    for c in np.unique(tl):
        ix = (tl == c)
        print(f'Class: {c}, Acc: {get_accuracy(cl[ix], tl[ix])}')

def get_color_match(query, predictions, scores=None):
    """
    :param predictions: a list of list. the i-th list is the list of predicted colors in i-th image
    :param scores: list of numpy vectors with the predicted score of the corresponding colors
    :param query: the list of colors  we are looking for
                  query can be a list (['red', 'blue'], ['red']) or a single color ('red')
    :return: the 'coincidence' between each prediction an the query, values from 0 to 1 .
            ie: proportion of colors in query also in prediction
    """
    
    # start_time = time.time()

    if type(query) is not list:
        query = [query]

    if scores is None:
        scores = [np.ones(len(pred)) for pred in predictions]


    keymap = [dict(set(zip(predictions[i], list(scores[i])))) for i in range(len(predictions))]
    matches = np.zeros(len(predictions))
    for ix, pred in enumerate(predictions):
        coincidences = set(pred).intersection(set(query))
        for c in coincidences:
            matches[ix] += keymap[ix][c]

    return matches / len(query)


def retrieve_img_by_color(test_imgs, predicted_test_colors, predicted_color_scores, query):
    """
    :param test_imgs: set of images to look in
    :param predicted_test_colors: list of list, predicted list of colors detected in each image
    :param predicted_color_scores: list og numpy vector with the predicted score of the corresponding colors
    :param query: the list of colors  we are looking for
                  query can be a list (['red', 'blue'], ['red']) or a single color ('red')
    :return: (imgs, coincidences, indexs):
                            imgs: numpy array of PxNxMx3, the sorted images by its coincidence with the query,
                            coincidences: numpy array of Px1 with the coincidence of the predicted colors and the query
                            indexs: the index from the original set of images that correspond to the returned list of imges
                                    imgs = test_imgs[indexs]
                             imgs are sorted by the probabity of the color names (predicted_color_scores)
    """
    query_prob = get_color_match(query, predicted_test_colors, predicted_color_scores)
    sort_index = query_prob.argsort()[::-1]
    sort_index = [x for x in sort_index if query_prob[x] > 0]
    return test_imgs[sort_index], query_prob[sort_index], sort_index




def retrieve_img_by_class(test_imgs, predicted_class_labels, predicted_class_scores, query):
    """
    :param test_imgs: set of images to look in
    :param predicted_class_labels: numpy vector of predicted classes per image
    :param predicted_class_scores: numpy vector with the predicted score of the corresponding class
    :param query: class to look for
    :return: (imgs, indexs), imgs: numpy array of PxNxMx3  P images of size NxMx3 whose predicted class is 'query'
                             indexs: indexes from the orignal test_imgs that are the output imgs
                             imgs = test_imgs[indexs]
                             imgs are sorted by the % of votes received in its class ( predicted_class_scores)
    """
    indexs = np.where(predicted_class_labels == query)
    # scores = np.array([predicted_class_scores[i] for i in indexs])
    sorted_scores = predicted_class_scores.argsort()[::-1]  # indexes of all sorted scores
    class_index = np.array([x for x in sorted_scores if x in indexs[0]])
    return test_imgs[class_index], class_index


def retrieve_combine(test_imgs, predicted_class_labels, predicted_class_scores,
                                predicted_color_labels, predicted_color_scores, query_class, query_color):
    """
    :param test_imgs: set of images to look in
    :param predicted_class_labels: numpy vector of predicted classes per image
    :param predicted_class_scores: numpy vector with the predicted score of the corresponding class
    :param predicted_color_labels: list of list, predicted list of colors detected in each image
    :param predicted_color_scores: list og numpy vector with the predicted score of the corresponding colors
    :param query_class:  class to look for
    :param query_color: the list of colors  we are looking for
                        query can be a list (['red', 'blue'], ['red']) or a single color ('red')
    :return: (imgs, indexs) imgs: list of images that are predicted as query_class, ordered by its
                                  color coincidence with query_color,
                            indexs: indexes of the images corresponding to the output, imgs = test_imgs[indexs]
    """
    new_imgs, class_index = retrieve_img_by_class(test_imgs, predicted_class_labels, predicted_class_scores, query_class)
    # new_imgs, coincidence, color_index = retrieve_img_by_color(new_imgs, np.array(predicted_color_labels)[class_index],
    #                                                            predicted_color_scores, query_color)
    new_imgs, coincidence, color_index = retrieve_img_by_color(new_imgs, np.array(predicted_color_labels)[class_index],
                                                               np.array(predicted_color_scores)[class_index], query_color)

    return new_imgs, class_index[color_index]


def check_correct_class(predict_labels, gt_labels, query):
    """
    :param predict_labels: list of predicted classes (sahpes) for a set of images
    :param gt_labels:  list of real (ground truth) classes for a set of images
    :param query: the class we are looking for
    :return: numpy array same lenght as predict_labels i-th element is 1 if the predicted  class and the
            real class are the query class, if not 0
    """
    return np.array([(1 if (predict_labels[i] == gt_labels[i] and gt_labels[i] == query)
                      else 0) for i in range(len(predict_labels))])


def check_correct_color(predict_labels, gt_labels, query):
    """
    :param predict_labels: list of predicted list of colors for a set of images
    :param gt_labels: list of real (ground truth) list of colors  for a set of images
    :param query:  the list of colors  we are looking for
                query can be a list (['red', 'blue'], ['red']) or a single color ('red')
    :return: numpy array same length as predict_labels. i-th element is 1 if at least one colors in the query color list
            is in the list of predicted colors and is in the list of real colors, if not 0
    """
    if type(query) is not list:
        query = [query]
    return np.array([(1 if (len(list(set(predict_labels[i]) & set(gt_labels[i]) & set(query))) > 0)
                      else 0) for i in range(len(predict_labels))])


def check_correct(predict_color_labels,  gt_color_labels, query_color, predict_class_labels, gt_class_labels, query_class):
    """
    given a set of class predictions, color predictions, real class labels, real color labels and class and color queries
    returns a vector of 0/1 if that specific entry matches the query on both, shape and color.
    :param predict_color_labels:
    :param gt_color_labels:
    :param query_color:
    :param predict_class_labels:
    :param gt_class_labels:
    :param query_class:
    :return:
    """
    return check_correct_color(predict_color_labels, gt_color_labels, query_color) * \
           check_correct_class(predict_class_labels, gt_class_labels, query_class)
