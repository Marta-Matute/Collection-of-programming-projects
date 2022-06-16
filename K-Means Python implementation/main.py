import pickle
import random

import matplotlib.pyplot as plt
import tqdm

from KNN import KNN
from Kmeans import *
from my_labeling import check_correct, check_correct_class, check_correct_color
from my_labeling import filter_colors, get_color_match, get_accuracy, print_acc_by_class
from my_labeling import precision_at_k
from my_labeling import retrieve_combine, retrieve_img_by_class, retrieve_img_by_color
from utils import colors
from utils_data import read_dataset, visualize_retrieval

if __name__ == '__main__':
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # plt.ion()

    ######################################################################################################
    ## KMeans
    ## Which is the performance evolution when changing k?  Will use 50 random test images (not enough). K up to 10
    ######################################################################################################
    num_im = 50
    indexs = random.sample(range(test_imgs.shape[0]), num_im)
    match = np.zeros((9, 1))
    for idx in tqdm.tqdm(indexs):
        for k in range(match.shape[0]):
            km = KMeans(test_imgs[idx], K=k + 2, options={'init': 'first', 'crop': 'croped'})  #
            km.fit()
            colors_km, scores = filter_colors(km, get_colors(km.centroids))
            match[k] += get_color_match(test_color_labels[idx], [colors_km])
    match /= num_im
    plt.figure()
    plt.plot(range(2, 2 + match.shape[0]), match, '-r')
    plt.ylim(0, 1)
    plt.xlabel('K in Kmeans')
    plt.ylabel('%  of real labels in predicted ')
    plt.title('effect of the  number of clusters')
    #plt.savefig('figures/1-kmeans_1.png'), plt.close()
    plt.show()

    ######################################################################################################
    # KNN
    # Which is the performance evolution when changing k?  the model are different % of all train images
    # Will use 50 random test images (not enough). K up to 10
    ######################################################################################################
    num_im = 50
    indexs = random.sample(range(test_imgs.shape[0]), num_im)
    per = [0.1, 0.3, 0.5, 0.7]
    match = np.zeros((9, len(per)))
    for p in range(len(per)):
        knn = KNN(train_imgs[:int(train_imgs.shape[0] * per[p])],
                  train_class_labels[:int(train_imgs.shape[0] * per[p])])
        for k in range(match.shape[0]):
            predicted_class_labels, predicted_class_scores = knn.predict(test_imgs[indexs], k + 2)
            match[k, p] += get_accuracy(predicted_class_labels, test_class_labels[indexs])
    plt.figure()
    plt.plot(range(2, 2 + match.shape[0]), match)
    plt.ylim(0, 1)
    plt.legend(per)
    plt.xlabel('k in KNN')
    plt.ylabel('accuracy')
    plt.title('effect of the size of the training set')
    plt.show()
    #plt.savefig('figures/2-knn_1.png'), plt.close()

    ######################################################################################################
    ## color prediction using best K (in that particular case with init='first' and
    #                                 whithinClassDistance and 20% of change in find_best_K)
    ######################################################################################################
    # set calculate_best_colors = True to run kmeans on all test images and save the predicted colors per image in a file
    # set calculate_best_colors = False to read the predictions from the file.
    calculate_best_colors = True
    if calculate_best_colors:
        predicted_color_labels = []
        predicted_color_scores = []
        for im in tqdm.tqdm(test_imgs):
            km = KMeans(im, options={'init': 'first', 'crop': 'croped'})  # , 'crop': 'croped'
            km.find_bestK(3)
            color_labels, color_scores = get_colors_and_probs(km)
            color_labels, color_scores = filter_colors(km, color_labels, color_scores)
            predicted_color_labels.append(color_labels)
            predicted_color_scores.append(color_scores)
            # uncomment next line if you want to see the kmeans result
            # visualize_k_means(km, im.shape)
        predicted_color_labels = np.array(predicted_color_labels)
        predicted_color_scores = np.array(predicted_color_scores)
        with open('predicted_best_color_labels.pkl', 'wb') as file:
            pickle.dump([predicted_color_labels, predicted_color_scores], file)
    else:
        with open('predicted_best_color_labels.pkl', 'rb') as file:
            predicted_color_labels, predicted_color_scores = pickle.load(file)

    ######################################################################################################
    ## class prediction using KNN  (in that particular case K=5, and all the images in the training set as data model)
    ######################################################################################################
    knn = KNN(train_imgs, train_class_labels)
    predicted_class_labels, predicted_class_scores = knn.predict(test_imgs, 3)
    print(f'Global accuracy on shape prediction: {get_accuracy(predicted_class_labels, test_class_labels)}')
    print_acc_by_class(predicted_class_labels, test_class_labels)

    ######################################################################################################
    # Retrieval
    ######################################################################################################

    # only color
    for query_color in colors:
        new_imgs, coincidence, indxs = retrieve_img_by_color(test_imgs, np.array(predicted_color_labels),
                                                             predicted_color_scores, [query_color])
        ok = check_correct_color(predicted_color_labels[indxs], test_color_labels[indxs], query_color)
        info = [f'Id:{idx}  {predicted_color_labels[idx]}\n real:{test_color_labels[idx]}' for coin, idx in
                zip(coincidence, indxs)]
        visualize_retrieval(new_imgs, 10, info=info, ok=ok,
                            title=f'Query: {query_color}')
        plt.savefig(f'figures/3-kmeans_{query_color}.png'), plt.close()

    # only class
    for query_class in classes:
        retrieved_imgs, indxs = retrieve_img_by_class(test_imgs, predicted_class_labels, predicted_class_scores,
                                                      query_class)
        ok = check_correct_class(predicted_class_labels[indxs], test_class_labels[indxs], query_class)
        if len(indxs) > 0:
            visualize_retrieval(retrieved_imgs, 10, info=test_class_labels[indxs], ok=ok,
                                title=f'Query: {query_class}')
        #plt.savefig(f'figures/4-knn_{query_class}.png'), plt.close()

    # combination of color and class
    for i in range(15):
        query_class = random.choice(classes)
        query_color = random.choice(colors)

        retrieved_imgs, indxs = retrieve_combine(test_imgs, predicted_class_labels, predicted_class_scores,
                                                 predicted_color_labels, predicted_color_scores, query_class,
                                                 [query_color])
        ok = check_correct(predicted_color_labels[indxs], test_color_labels[indxs], query_color,
                           predicted_class_labels[indxs], test_class_labels[indxs], query_class)
        if len(retrieved_imgs) > 0:
            info = [f'Id:{idx}  {test_class_labels[idx]}\n real:{test_color_labels[idx]}' for idx in indxs]
            visualize_retrieval(retrieved_imgs, 15, info=info, ok=ok,
                                title=f'Query: {query_color + " " + query_class}')
        #plt.savefig(f'figures/5-combine_{i}.png'), plt.close()

    # images as query
    for i in range(15):
        idx = random.choice(range(test_imgs.shape[0]))
        query_class = test_class_labels[idx]
        query_color = test_color_labels[idx]
        
        retrieved_imgs, indxs = retrieve_combine(test_imgs, predicted_class_labels, predicted_class_scores,
                                                 predicted_color_labels, predicted_color_scores, query_class,
                                                 query_color)
        ok = check_correct(predicted_color_labels[indxs], test_color_labels[indxs], query_color,
                           predicted_class_labels[indxs], test_class_labels[indxs], query_class)
        if len(retrieved_imgs) > 0:
            info = [f'Id:{idx}  {test_class_labels[idx]}\n real:{test_color_labels[idx]}' for idx in indxs]
            visualize_retrieval(retrieved_imgs, 15, info=info, ok=ok,
                                title=f'Query: {f"{query_color}" + " " + query_class}', query=test_imgs[idx])
            #plt.savefig(f'figures/6-combine_img_{i}.png'), plt.close()

    Ks = [1, 5, 10, 15, 20]
    PatK = np.zeros(len(Ks))
    for idx in range(test_imgs.shape[0]):
        query_class = test_class_labels[idx]
        query_color = test_color_labels[idx]

        retrieved_imgs, indxs = retrieve_combine(test_imgs, predicted_class_labels, predicted_class_scores,
                                                 predicted_color_labels, predicted_color_scores, query_class,
                                                 query_color)
        ok = check_correct(predicted_color_labels[indxs], test_color_labels[indxs], query_color,
                           predicted_class_labels[indxs], test_class_labels[indxs], query_class)
        for i, k in enumerate(Ks):
            PatK[i] += precision_at_k((query_class, query_color),
                                      (test_class_labels[indxs[:k]], test_color_labels[indxs[:k]]))
    PatK /= test_imgs.shape[0]
    plt.figure()
    plt.plot(Ks, PatK, '-r')
    plt.xlabel('K')
    plt.ylabel('Precission at K')
    plt.title('Retrieval Performance')
    plt.show()
    #plt.savefig(f'figures/7-Patk.png'), plt.close()