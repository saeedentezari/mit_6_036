#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Auto, reviews, and mnist data analysis
======================================

Auto data analysis
------------------

We consider various feature function choices for representing numerical data, such as raw,
    standardized, or one-hot representing. Along with finding the best algorithm and its hyperparameter
    in terms of accuracy of classification.
An interpretation of best obtained classifier has been presented.


Reviews data analysis
---------------------

Then we work with textual data. We try to represent them (in bag-of-words perspective)
    and find the best bow representation by the options of:
        filter stopwords, count tokens, unigram/bigram.
Then again we try to find the best algorithm and its hyperparameter.
By having these, we can get the best classification coefficients and look up the biggest ones,
    which corresponds to most negative and most positive words in terms of sentiment.


MNIST data analysis
-------------------

The image data here they come. We consider some feature extractions out of images such as:
    row average, column average, and top/bottom average representations.
Then we look at several digits classification two by two and some interpretation has been provided.


Run the file and enjoy it!
"""


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import my_code_for_hw3_part2 as hw3
from my_code_for_hw3_part1 import rv
from pprint import pprint
import time


__author__ = "Saeed Entezari"
__email__ = "s.entezari144@gmail.com"


directory_data = Path('~/mit_6_036/w3_feature_representation/lab3_data')


##########################################################################################
# Handy write and read functions
##########################################################################################

import pickle

def write_on_file(variable, filename):
    '''
    Write the variable value into a file.
    '''
    with open(filename, 'wb') as fhand:
        pickle.dump(variable, fhand)
    print(f'\nvariable given has dumped on file {filename}.')

def load_from_file(filename):
    '''
    Load the variable value from the file.
    '''
    with open(filename, 'rb') as fhand:
        variable = pickle.load(fhand)
    print(f'\nfile {filename} loaded to variable.')
    return variable

print("""\n
##########################################################################################
# Auto Data Analysis
##########################################################################################
""")

path_auto_data = directory_data / 'auto-mpg.tsv'
auto_data_dict = hw3.load_auto_data(path_auto_data)


# learners
learners = [hw3.perceptron, hw3.averaged_perceptron]
# feature choices
# raw choices
featfuncs1 = [('cylinders', hw3.raw),
              ('displacement', hw3.raw),
              ('horsepower', hw3.raw),
              ('weight', hw3.raw),
              ('acceleration', hw3.raw),
              ('origin', hw3.raw)]
# advance choices
featfuncs2 = [('cylinders', hw3.one_hot),
              ('displacement', hw3.standard),
              ('horsepower', hw3.standard),
              ('weight', hw3.standard),
              ('acceleration', hw3.standard),
              ('origin', hw3.one_hot)]
# set of just two features
featfuncs3 = [('cylinders', hw3.one_hot),
              ('weight', hw3.standard)]
featchoices = [featfuncs1, featfuncs2, featfuncs3]
# algorithm's number of iterations hyperparameters
T = [1, 10, 50]

print('\nHere we consider 3 feature function choices as:')
for n, featfuncs in enumerate(featchoices):
    print('\nfeature_choice_'+str(n))
    # pprint(list(map(lambda featfunc: (featfunc[0], featfunc[1].__name__), featfuncs)))
    pprint([(feat, func.__name__) for feat, func in featfuncs])

print("\nAnd 3 T's as:", T)
print("\nBy performing on two perceptron and averaged perceptron algorithms.")
input('\nContinue?')


print('\n------------------------ frequency histogram of raw features ---------------------------')


print("""\nFirst of all let's look at frequency histogram of all the features on their dimensions
    to see if there is any feature which separates the datapoints.""")
auto_data_raw, auto_labels_raw = hw3.auto_data_and_labels(auto_data_dict, featfuncs1)
fig, axes = plt.subplots(auto_data_raw.shape[0])
for featnum, axis in enumerate(axes):
    axis.hist(auto_data_raw[featnum, auto_labels_raw[0,:] > 0], label='positive mpg label')
    axis.hist(auto_data_raw[featnum, auto_labels_raw[0,:] < 0], label='negative mpg label')
    axis.legend()
    axis.set_xlabel(featfuncs1[featnum][0])
    axis.set_ylabel('freq. hist.')
plt.tight_layout()
print('\n--- Showing frequency histograms of raw features...')
print('\nYou can easily see the two features "cylinders" and "weight" can separate the data.')
plt.pause(1)
input('\nContinue?')



print('\n--------------------------------- accuracy analysis table ------------------------------------')


def auto_accuracy_table(auto_data_dict, featchoices, learners, T, k=10, **kwargs):
    '''
    Compute k-fold cross-validation accuracies for each combination of feature choices, learners, and T's.
        Then returns the result as a multi-index dataframe.

    Params:
        auto_data_dict (list of dicts): See hw3.load_auto_data function.
        featchoices (list of featfuncs): each featfunc is a list of (feature name, feature functions)
        learners (list of learner functions): it can be [perceptron, averaged_perceptron]
        T (list of ints): number of iteration hyper-parameters for perceptron and average_perceptron
        k (int, optional = 10): k-fold cross-validation

    Returns:
        a multi-index dataframe of accuracies
    '''
    seed = kwargs.get('seed', None)
    # loop over feature choices, learners and T's
    # and collect all the accuracies for each combination
    table = []
    for featchoice in featchoices:
        # prepare data and labels for each featchoices
        auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_dict, featchoice)
        for learner in learners:
            row = []
            for it in T:
                # get the accuracy
                accuracy = hw3.xval_learning_alg(learner, auto_data, auto_labels, k, learner_params={'T': it}, seed=seed)
                row.append(accuracy)
            table.append(row)
    # make a multi-index dataframe and return it
    featchoices_str = ['feature_choice_'+str(i+1) for i in range(len(featchoices))]
    learners_str = [learner.__name__ for learner in learners]
    indices = pd.MultiIndex.from_product([featchoices_str, learners_str])
    return pd.DataFrame(table, index=indices, columns=T)

# get the analysis table of accuracies
print('\nPlease wait a few seconds to get the accuracy table...')
acc_table = auto_accuracy_table(auto_data_dict, featchoices, learners, T, seed=0)
print('\nAccuracy table:\n', acc_table)

# plot accuracies for each feature choice
fig, axes = plt.subplots(acc_table.index.unique(level=0).size)
for axis, featchoice in zip(axes, acc_table.index.unique(level=0)):
    acc_table.loc[featchoice].T.plot(kind='bar', xlabel='T', ylabel='accuracy', title=featchoice, ax=axis)
print('\n--- Showing accuracies in bar plot...')

print("""\nConclusion:
    In terms of accuracy, averaged perceptron works better than perceptron in every cases.
    The hyperparameter T = 1 gives a sensible accuracy,
    And the second and third feature choices has very closed accuracies.""")
plt.pause(1)
input('\nContinue?')


print('\n----------------------------- best classifier coefficients -----------------------------')

print("""
It has been determined that the best
    algorithm: averaged_perceptron
    feature choice: feature_choice_2 (or feature_choice_1 which is more time efficient)
    T: 1
    
    Now we want to find coefficients of the best classifier,
    to see which features have the most impact on the output prediction.""")

# construct best classifier using all the data
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_dict, featfuncs2)
best_th, best_th_0 = hw3.averaged_perceptron(auto_data, auto_labels, params={'T': 1})
# make indices' name to know each coefficient of the best classifier
th_indices = []
for feat, func in featfuncs2:
    if func == hw3.one_hot:
        entries = list(set(datum[feat] for datum in auto_data_dict))
        th_indices.extend([feat + ' (one-hot)']*len(entries))
    else:
        th_indices.append(feat)
# print out the best classifier coefficients
print('\nBest Classifier coeffs:\n', pd.Series(best_th[:,0], index=th_indices))

print("""\nConclusion:
    We can see that the "weight" has the biggest coefficient in absolute value.
    But we should be careful about the scales:
        weight is a standardized feature, which means its scales can be considered as its variance, 1.
        one-hot features has a smaller scale, because they are restricted in range 0 to 1, and have a scale
        about half of the weight's scale.
    Regardless of scale, how can we measure the impact of an one-hot feature? while they have multiple coefficients?
        I would rather to take the maximum absolute value of the coefficients as a measure.""")

input('\nContinue?')


print("""\n
##########################################################################################
# Review Data Analysis
##########################################################################################
""")

path_review_data = directory_data / 'reviews.tsv'
review_data_dict = hw3.load_review_data(path_review_data)
review_data_dict_small = hw3.load_review_data(path_review_data, maxline=1000)


# An Explanation on zip() function
# ================================

# >>> pow = [0, 1, 2, 3]
# >>> exp = [1, 2, 4, 8]

# # packing
# >>> pow_exp = zip(pow, exp)
# >>> pow_exp
#     <zip object at ...>
# >>> list(pow_exp)
#     [(0, 1), (1, 2), (2, 4), (3, 8)]
# >>> list(pow_exp)
#     []            # zip object is an iterator which yields once


# >>> pow_exp = zip(pow, exp)
# # now suppose we want to unzip pow_exp, or any other list of iterables with the same length
# # unpacking
# >>> pow, exp = zip(*pow_exp)
# >>> pow
#     (0, 1, 2, 3)
# >>> exp
#     (1, 2, 4, 8)
# >>> exp
#     (1, 2, 4, 8)      # notice it's not an iterator, the result of unpacking stored in pow and exp variables


def prepare_bow_data_labels(review_data_dict, filter_stopwords = False, count_tokens = False, ngram = 1,
                                return_dictionary = False):
    '''
    Prepare bag-of-words data and labels by parameters in dictionary construction and bow feature extraction.

    Params:
        review_data_dict (list of dicts): see hw3.load_review_data function
        filter_stopwords (bool, optional = False): will be given to hw3.bag_of_words function
        count_tokens (bool, optional = False): will be given to hw3.extract_bow_feature_vector function
        ngram (int, optional = 1): will be given to hw3.bag_of_words function
        return_dictionary (bool, optional = False): if True, the function return dictionary in addition to the data and labels

    Returns:
        review data in bow representation (#tokens x #reviews array)
        review labels (1 x #reviews array)
        if return_dictionary: dictionary (dict of token -> tokenid)
    '''
    # replace text and sentiment of reviews in two separate lists
    review_texts, review_labels_list = zip(*[(review['text'], review['sentiment']) for review in review_data_dict])

    # create the bag-of-words dictionary for all the words
    dictionary = hw3.bag_of_words(review_texts, filter_stopwords, ngram)

    # bow representation of the texts data by the help of dictionary obtained
    review_bow_data = hw3.extract_bow_feature_vector(review_texts, dictionary, count_tokens=count_tokens)
    review_labels = rv(review_labels_list)

    # return
    if return_dictionary: return review_bow_data, review_labels, dictionary
    else: return review_bow_data, review_labels



print('\n-------------------------- which feature representation chcoice? -----------------------')

print("""\nFeatures can be represented as a combination of some options:
    filter stopwords: False or True
    count tokens: False or True
    ngram: 1 (unigram) or 2 (bigram)""")
print("""\nBefore picking the best algorithm and T, we want to find out which feature representation choice
    is more effective in accuracy for class of perceptron algorithms.
    (Notice that the efficieny of the feature representation is somehow dependent on the algorithm which we give it to.)
    For which, we consider the count tokens and filter stopwords as True False parameters,
    and also different n-grams corresponding to unigram and bigram.""")

# set simple T and learner for a quick obtaining of the best feature representation choice
T = 1
learner = hw3.averaged_perceptron
print(f"""\nFirst we train the averaged perceptron with T = 1 for a quick comparison about
    different combinations of count tokens and filter stopwords.""")

input('\nContinue?')


print('\n---------------------------- count tokens and filter stopwords -------------------------')

accuracies = np.zeros((2, 2))
print('\nPlease wait a few minutes to process the result!')
# compute the accuracies for filter_stopwords and count_words each False and True
for f, filter_stopwords in enumerate([False, True]):
    for c, count_tokens in enumerate([False, True]):
        review_bow_data, review_labels = prepare_bow_data_labels(review_data_dict, 
                                            filter_stopwords=filter_stopwords, count_tokens=count_tokens)
        accuracies[f, c] = hw3.xval_learning_alg(learner, review_bow_data, review_labels, learner_params={'T': T}, seed=0)

# plot a heatmap to see the results
fig, ax = plt.subplots()
matrix = ax.imshow(accuracies)
ax.set_xticks(np.arange(2), labels=[False, True])
ax.set_yticks(np.arange(2), labels=[False, True])
ax.set_xlabel('count tokens')
ax.set_ylabel('filter stopwords')
for i in range(2):
    for j in range(2):
        ax.text(j, i, round(accuracies[i, j], 3))
ax.set_title(f'accuracy for learner {learner.__name__} and T = {T}')
fig.colorbar(matrix)
print('\n--- Showing the acccuracies...')
plt.pause(1)

print(f"""\nConclusion:
    It is interesting that the best performance is for count_tokens = False and filter_stopwords = False.
    Notice that the learning algorithm is the {learner.__name__}.
    By filter_stopwords = False we have some extra dimensions which can be helpful to separability of datapoints.
    By count_tokens = False we have all the datapoints in a multi-dimensional cube by length 1, and it is more scalable.
    Of course, if we increase the T value, count_tokens = True can be helpful to separability and cosequently to accuracy.
    
    So the count_tokens = False, filter_stopwords = False is now picked up!
    They are set to default arguments of the prepare_bow_data_labels function.""")

input('\nContinue?')


print('\n--------------------------------------- n-gram -----------------------------------------')

print(f"""\nWhich is better? unigram or bigram?
    To know about, we consider a smaller version of the reviews (because in bigram case,
    the number of unique tokens grows much more faster than the unigram case).
    We perform {learner.__name__} with T = {T} and get the shapes of the data representations and
    the accuracies.\n""")

for ngram in [1, 2]:
    print(f'-- ngram = {ngram}:')
    review_bow_data, review_labels = prepare_bow_data_labels(review_data_dict_small, ngram=ngram)
    print('\treview bow data shape:', review_bow_data.shape, '\t (#tokens, #reviews)')
    print('\treview labels shape:', review_labels.shape)
    accu = hw3.xval_learning_alg(learner, review_bow_data, review_labels, learner_params={'T': T}, seed=0)
    print(f'\taccuracy of learner {learner.__name__} with T = {T}:\n\t', round(accu, 3), '\n')

print("""\nConclusion:
    In an equal low T, the unigram performs better than the bigram.
    The problem of bigram is the high-dimensionality of it and in low T's it can not perform well.
    Maybe if we could increase the number of iterations T, then the bigram would give us a better performance.

    The unigram is set to default argument.
    
    Now that we have the best feature representation choice, we can try to find out about the best
    algorithm and its hyperparameter on this dataset.""")

input('\nContinue?')


print('\n---------- best algorithm and hyperparameter based on accuracy and accuracy per runtime ------------')

review_bow_data, review_labels = prepare_bow_data_labels(review_data_dict)

print("""\nWith the feature representation choosed for class of perceptron algorithms,
    we want to know which algorithm and which hyperparameter T has the best efficiency in
    expected accuracy and runtime simultaneously.""")

learners = [hw3.perceptron, hw3.averaged_perceptron]
T = [1, 10, 50]

print('\nPlease wait a few minutes to process the result!')
# the first dimension of the result with length 2 is corresponding to
# accuracies and accuracies per runtime variables respectively
results = np.zeros((2, len(learners), len(T)))
for l, learner in enumerate(learners):
    for t, hyper in enumerate(T):
        ti = time.time()
        accuracy = hw3.xval_learning_alg(learner, review_bow_data, review_labels, learner_params={'T': hyper}, seed=0)
        tf = time.time()
        results[0, l, t] = accuracy             # accuracy
        results[1, l, t] = accuracy / (tf-ti)   # accuracy / runtime

# plot heatmaps to see the results
fig, axes = plt.subplots(2)
for var, axis in enumerate(axes):   # var = 0 -> accuracy, var = 1 -> accuracy / runtime
    mapple = axis.imshow(results[var])
    # title, labels and ticks
    axis.set_title('Accuracy') if var == 0 else axis.set_title('Accuracy per Runtime [1/second]')
    axis.set_xlabel('T')
    axis.set_ylabel('learners')
    axis.set_yticks(np.arange(len(learners)), labels = [learner.__name__ for learner in learners])
    axis.set_xticks(np.arange(len(T)), labels = T)
    # show the values of the heatmap by text
    for l in range(len(learners)):
        for t in range(len(T)):
            axis.text(t, l, round(results[var, l, t], 3))
    # colorbar
    fig.colorbar(mapple, ax=axis)
print('\n--- Showing the accuracies and accuracies per runtime...')
plt.pause(1)

print("""\nConclusion:
    From the heatmap it is clear that the averaged perceptron has the better expected accuracy in every T's.
    But the better accuracy per runtime is related to the perceptron.
    By considering the accuracy as primary key and the accuracy per runtime as secondary key,
    we choose averaged perceptron and the best accuracy for averaged perceptron is related to T = 10,
    however, the accuracy per runtime in T = 10 is not as bad as T = 50,
    and it is acceptable in comparison with T = 1.
    
    So the best set of algorithm and hyperparameter is
    averaged perceptron and T = 10.""")

input('\nContinue?')


print('\n-------------------------- 10 most positive and negative words ---------------------------')

print("""\nWe want to find 10 most positive and negative words,
    the words that contribute most to a positive or negative prediction.
    We construct the best classifier using all data and pick the 10 largest positive and negative
    coefficients of the best classifier which correspond to 10 most positive and negative words.""")

# prepare the best feature representation obtained for averaged perceptron
review_bow_data, review_labels, dictionary = prepare_bow_data_labels(review_data_dict, return_dictionary=True)
# learn averaged perceptron by T = 10, using all data to get the best classifier
th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, params={'T': 10})

# pack the coordinates by their indices
index_coor = list(zip(range(len(th)), th[:,0]))
# sort the packed variable by the coordinate value as key
index_coor.sort(key=lambda tup: tup[1])
# so now index_coor is sorted

# get the reverse dictionary, tokenid -> token
rev_dictionary = hw3.reverse_dict(dictionary)

# 10 most positive words
print('\n10 words that contribute most to a positive prediction:\n\t',
    [rev_dictionary[tokenid] for tokenid, coor in index_coor[-10:]])

# 10 most negative words
print('\n10 words that contribute most to a negative prediction:\n\t',
    [rev_dictionary[tokenid] for tokenid, coor in index_coor[:10]])

input('\nContinue?')



print('\n---------------------------- most positive and negative reviews ---------------------------')

print('''\nNow we want to find the most positive and negative "reviews".
    That is, ones with the most positive and negative distance to the hyperplane.''')

# prepare the data and labels by hand, not by the prepare_bow_data_labels function,
# to be sure of having the review texts list in the same order of review_bow_data array (second dimension)
review_texts, review_labels_list = zip(*[(review['text'], review['sentiment']) for review in review_data_dict])
dictionary = hw3.bag_of_words(review_texts)
review_bow_data = hw3.extract_bow_feature_vector(review_texts, dictionary)
review_labels = rv(review_labels_list)

# obtain the best classifier using all data as train data
th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, params={'T': 10})

# compute the distance of all datapoints to the separator
dists = hw3.distance_to_separator(review_bow_data, th, th0)

# sort the distances
index_dist = list(zip(range(dists.shape[1]), dists[0]))
index_dist.sort(key=lambda tup: tup[1])

# most positive review
print('\nMost positive review (has the most positive distance to the classifier):\n\n\t',
    review_texts[index_dist[-1][0]])    # the index of the last element of index_dist

# most negative review
print('\nMost negative review (has the most negative distance to the classifier):\n\n\t',
    review_texts[index_dist[0][0]])     # the index of the first element of index_dist

input('\nContinue?')



print("""\n
##########################################################################################
# MNIST Data Analysis
##########################################################################################
""")


# load mnist data dictionary for all digits
mnist_data_all = hw3.load_mnist_data(range(10))
"""
hw3.load_mnist_data function returns a dictionary formatted as follows:

{
    0: {
        'images': [(mxn image array of 0), ...],
        'labels': 1xlen(0 images) array of digit 0
    }

    1: {
        'images': [(mxn image array of 1), ...],
        'labels': 1xlen(1 images) array of digit 1
    }

    ...

    9: ...
}

See hw3.load_mnist_data and hw3.load_mnist_single functions for more details.
"""

def prepare_two_digits_data_labels(mnist_data_all, dig1, dig2):
    '''
    Extract images for two digits given, from mnist data dictionary
        and put all images in a 3D data array which its first axis is associated with images.
        Consider their labels value as -1 and +1 for the first and second digits respectively
        and create labels array correspondingly in order of data array images.

    Params:
        mnist_data_all (dict): to see the format, see hw3.load_mnist_data function.
        dig1, dig2 (int): digits given to extract their images and assign labels.

    Returns:
        3D data array of the shape (n_samples, m, n)
            which each image is a mxn array
            and n_samples = # samples for dig1 + # samples for dig2.
        2D labels array of the shape (1, n_samples)
            with -1 and +1 labels for dig1 and dig2 respectively
            and the labels value is corresponding to images in data array.
    '''
    # a list of images (each image as 28x28 array) for first digit
    imglst1 = mnist_data_all[dig1]['images']
    # and for second digit
    imglst2 = mnist_data_all[dig2]['images']
    # consider -1 labels for first digit as a 1xlen(imglst1) array
    labels1 = np.repeat(-1, len(imglst1)).reshape(1, -1)
    # and +1 labels for second digit
    labels2 = np.repeat(1, len(imglst2)).reshape(1, -1)

    # data is a 3D array of images
    # which goes into the feature computation functions later
    data = np.stack(imglst1 + imglst2)    # stack two list of images along a new axis
    # labels can directly go into the perceptron algorithm
    labels = np.hstack((labels1, labels2))

    return data, labels



# ---------------------------- feature extraction functions from data -----------------------------


def raw_mnist_features(data):
    '''
    Reshape 3D data array of the shape (n_samples, m, n) into
        2D data array of the shape (m*n, n_samples).

        In fact, it ravels each image (first axis) of the data array
        and put it in a column of a 2D array returned,
        to make raw data ready to give to perceptron algorithm.
    '''
    n_samples, m, n = data.shape
    return data.reshape(n_samples, m*n).T


def row_average_features(data):
    '''
    Get the row average of each mxn image (first axis) of the data array (n_samples, m, n)
        and put it in the columns of the returned array of the shape (m, n_samples),
        to extract row average features out of images
        and make them in ready shape to give to perceptron algorithm.
    '''
    return np.mean(data, axis=-1).T


def col_average_features(data):
    '''
    Get the col average of each mxn image (first axis) of the data array (n_samples, m, n)
        and put it in the columns of the returned array of the shape (m, n_samples),
        to extract col average features out of images
        and make them in ready shape to give to perceptron algorithm.
    '''
    return np.mean(data, axis=1).T


def top_bottom_features(data):
    '''
    Get the top (exclusive half) and bottom (inclusive half) averages
        of each mxn image (first axis) of the data array (n_samples, m, n)
        and put it in the columns of the returned array of the shape (2, n_samples).
    '''
    n_samples, m, n = data.shape
    # slice the top (exclusive) half of the data
    # and get the average along all axes but the first
    top_means = np.mean(data[:, :m // 2], axis=(1, 2))
    # slice the bottom (inclusive) half of the data
    # and get the average along all axes but the first
    bot_means = np.mean(data[:, m // 2:], axis=(1, 2))
    # stack top and bottom averages vertically
    # so each colomn of the returned array is of length 2, top and bottom averages
    return np.vstack((top_means, bot_means))


vsdigs = [(0, 1), (2, 4), (6, 8), (9, 0), (2, 7), (0, 8), (6, 9), (6, 7), (2, 3), (4, 5), (4, 7)]
print('\nDigits we want to compare their classification accuracies with perceptron algorithm is:')
for dig1, dig2 in vsdigs:
    print(f'\t{dig1} vs. {dig2}')

input('\nContinue?')



print('\n-------------------------- use raw features as baseline accuracies ---------------------------')


print("""\nBy raw features we mean each image vectorized by flattening its array.\n""")

raw_accs = {}
for dig1, dig2 in vsdigs:
    data, labels = prepare_two_digits_data_labels(mnist_data_all, dig1, dig2)
    acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)
    raw_accs[f'{dig1} vs. {dig2}'] = [acc]
    print(f'-- accuracy for {dig1} vs. {dig2} is:\t {round(acc, 3)}')

input('\nContinue?')



print('\n----------------- row, col, top/bottom feature representations accuracies --------------------')


print("""\nRow averages feature representation gets the average values of the rows of image as vectorization of it.
    And so for column averages.
    Top/bottom averages represents each image as a vector of length 2, average of top half values of the image as the first
    and average of bottom as the second dimension of it.\n""")

# a dictionary for storing feature extracted accuracis
# 'dig1 vs. dig2' -> [row, col, top/bottom] feature representation accuracies
print('Accuracies are shown in the order [row, col, top/bottom]\t[raw]')
fext_accs = {}
for dig1, dig2 in vsdigs:
    data, labels = prepare_two_digits_data_labels(mnist_data_all, dig1, dig2)
    rowacc = hw3.get_classification_accuracy(row_average_features(data), labels)
    colacc = hw3.get_classification_accuracy(col_average_features(data), labels)
    topbotacc = hw3.get_classification_accuracy(top_bottom_features(data), labels)
    fext_accs[f'{dig1} vs. {dig2}'] = [rowacc, colacc, topbotacc]
    print(f'-- accuracies for {dig1} vs. {dig2} is:\t{[round(rowacc, 3), round(colacc, 3), round(topbotacc, 3)]}', end='\t')
    print(list(map(lambda x: round(x, 3), raw_accs[f'{dig1} vs. {dig2}'])))


print("""\nConclusion:
    Almost all the raw accuracies are higher than any one of extracted features accuracies,
        except for some, like the 6 vs. 9, or 6 vs. 8, with a slight differences in accuracy.
        It can be because of "the round shape" of digits may be confusing for the perceptron
        which makes it hard to distinguish between them.
        meanwhile the row extracted features can help to resolve this ambiguity in detecting the tails
        for this kind of digits in classifying them.
    Between the extracted features, usually the row is the best one,
        except one case I noticed, 0 vs. 8, which have the very same shape in row averages.
    For the col and top/bottom comparison, its very dependent on the digits to classify.
        For example, in classifying 6 vs. 8, the top/bottom is more helpful,
        but in 0 vs. 8, the col averages has better result.
    If we set aside 0 vs. 1, because it is very distinguishable in every feature extraction cases,
        and also the row and raw features for the high accuracy results,
        and we exclude 4 vs. 5 because of bad accuracies in col and top/bottom averages,
        then the best accuracy for col averages is corresponding to 9 vs. 0, and the worst is for 4 vs. 7,
        and the best and worst for top/bottom representations is for 6 vs. 7 and 0 vs. 8 respectively.""")

input('\nDone?')