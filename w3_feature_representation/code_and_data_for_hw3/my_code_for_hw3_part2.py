#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Loading and representing auto (numerical), reviews (textual), and mnist (visual) data
-------------------------------------------------------------------------------------

This file contains tools of:
    loading auto data from a tsv file and doing some choices of representing numerical data like:
    standardiz, one-hot, or leave them to be raw.

    loading reviews data from a tsv file and representing textual data in the bag-of-words form.

    loading handwritten digits from png files and formatting them into a dataset.
"""


import csv
from pathlib import Path
import numpy as np
from pprint import pprint


__author__ = "Saeed Entezari"
__email__ = "s.entezari144@gmail.com"



directory_data = Path('~/mit_6_036/w3_feature_representation/lab3_data')


###############################################################################
# Perceptron and Average Perceptron: Algorithm and Evaluation
###############################################################################

def prediction(x, th, th_0):
    '''
    Prediction of label of datapoints x (dxn array)
        with respect to separator specified by parameters
        theta (dx1 array) and theta_0 (1x1 array).
    
    Returns (1xn array of) sign (int in {-1, 0, 1})
    '''
    return np.sign(np.dot(th.T, x) + th_0)

def norm(th):
    '''
    Norm of theta (dx1 array)
    '''
    return np.sqrt(np.sum(th*th))

def distance_to_separator(x, th, th_0):
    '''
    Returns distance to separator for datapoints x.

    Parameters:
        x (dxn array)
        th (dx1 array)
        th_0 (1x1 array)
    
    Returns:
        (1xn array of) distances (float) for datapoints.
    '''
    return (np.dot(th.T, x) + th_0) / norm(th)

def perceptron(datapoints, labels, params = {}, hook = None):
    '''
    Perform perceptron algorithm for datapoints with labels,
        by going through datapoints T times.

    Parameters:
        datapoints (dxn array)
        labels (1xn array): values in {-1, 0, +1}
        params (dict, optional = {}): extra parameters to algorithm,
            such as T, the number of iterations through dataset.
        hook (func, optional = None): a function for plotting separator, if given.

    Returns:
        final th (dx1 array)
        final th_0 (1x1 array)
    '''
    # if T not in params, set it to 50
    T = params.get('T', 50)
    # initialization
    d, n = datapoints.shape
    th = np.zeros((d, 1))
    th_0 = np.zeros((1, 1))
    # iteration
    for iter in range(T):
        for i in range(n):
            y_i = labels[0, i]
            x_i = datapoints[:, i:i+1]
            if y_i * prediction(x_i, th, th_0) <= 0:   # mistake and update
                th = th + y_i * x_i
                th_0 = th_0 + y_i
                if hook: hook((th, th_0), i)
    return th, th_0


def averaged_perceptron(datapoints, labels, params={}, hook=None):
    '''
    Average perceptron algorithm. See perceptron function.

    Parameters:
        datapoints (dxn array)
        labels (1xn array)
        params (dict, optional = {}): of extra parameters to the algorithm,
            such as T, the number of iteration through dataset.
        hook (func, optional = None): for plotting separator and datapoints.
    
    Returns:
        average of all theta's in all iterations of algorithm (T*n iterations)
        average of all theta_0's such as theta's
    '''
    # if T not in params, default to 100
    T = params.get('T', 100)
    # initialization
    d, n = datapoints.shape
    th = np.zeros((d, 1))
    th_0 = np.zeros((1, 1))
    thSum = th[:]
    th_0Sum = th_0[:]
    # iteration
    for iter in range(T):
        for i in range(n):
            y_i = labels[0, i]
            x_i = datapoints[:, i:i+1]
            if y_i * prediction(x_i, th, th_0) <= 0:    # mistake and update
                th = th + y_i * x_i
                th_0 = th_0 + y_i
                if hook: hook((th, th_0))
            thSum = thSum + th
            th_0Sum = th_0Sum + th_0
    thAvg = thSum / (T*n)
    th_0Avg = th_0Sum / (T*n)
    if hook: hook((thAvg, th_0Avg))
    return thAvg, th_0Avg


def score(datapoints, labels, th, th_0):
    '''
    Calculate the ratio of correct predictions on dataset
        based on separator th, th_0.

    Parameters:
        datapoints (dxn array)
        labels (1xn array)
        th (dx1 array), th_0 (1x1 array or float)
    
    Returns:
        ratio (float) between 0 and 1
    '''
    d, n = datapoints.shape
    ncorrect = np.sum(prediction(datapoints, th, th_0) == labels)
    return ncorrect / n


def eval_classifier(learner, data_train, labels_train, data_test, labels_test, learner_params = {}):
    '''
    Perform classifier on training dataset and evaluate the score of outcome hypothesis on test dataset.

    Parameters:
        learner (func): learning algorithm, perceptron or averaged perceptron.
        data_train (dxntrain array)
        labels_train (1xntrain array)
        data_test (dxntest array)
        labels_test (1xntest array)
        learner_params (dict, optional = {}): parameters to pass to learner

    Returns:
        test_score (float) between 0 and 1
    '''
    # run learner with training dataset to get learned parameters
    learned_th, learned_th_0 = learner(data_train, labels_train, learner_params)
    # evaluate the obtained separator on test dataset
    test_score = score(data_test, labels_test, learned_th, learned_th_0)
    # return ratio of correctness of prediction with respect to obtained separator in test dataset
    return test_score


def shuffle(datapoints, labels, seed = None):
    '''
    Shuffle datapoints (dxn array) and its associated labels (1xn array) column-wise.
        seed can be given.

    Returns:
        Shuffled datapoints and labels.
    '''
    d, n = datapoints.shape
    indices = list(range(n))
    # make shuffled indices
    if seed is not None: np.random.seed(seed)
    np.random.shuffle(indices)
    # return shuffled dataset from shuffled indices
    return datapoints[:, indices], labels[:, indices]


def xval_learning_alg(learner, datapoints, labels, k=10, learner_params = {}, **kwargs):
    '''
    Cross-validation evaluation of a learning algorithm.
        Divide dataset into k chunks and get each of them as
        testing dataset and the rest for training dataset in each evaluation step.
        Return the average of test scores along the cross-validation.

    Params:
        learner (func): learning algorithm, perceptron or averaged perceptron
        datapoints (dxn array)
        labels (1xn array)
        k (int): number of chunks to divide the data
        learner_params (dict, optional = {}): parameters to pass to learner
            NOTE: Alternatively we could construct the learner and set its hyperparameter T
                by lambda function in calling the xval_learning_alg.


    Returns:
        average of test scores along the cross-validation process (float between 0 to 1)
    '''
    seed = kwargs.get('seed', None)
    # shuffle the dataset before start
    datapoints, labels = shuffle(datapoints, labels, seed)
    # divide data into k chunks in a list
    datapoints_chunks = np.array_split(datapoints, k, axis=1)
    labels_chunks = np.array_split(labels, k, axis=1)
    # iteration k times for picking datasets and evaluations
    scores = []
    for i in range(k):
        # pick i'th chunk as test dataset
        datapoints_test, labels_test = datapoints_chunks[i], labels_chunks[i]
        # to get the training dataset, add the rest chunks together and np.concatenate them along axis=1 which was splitted
        datapoints_train = np.concatenate(datapoints_chunks[:i]+datapoints_chunks[i+1:], axis=1)
        labels_train = np.concatenate(labels_chunks[:i]+labels_chunks[i+1:], axis=1)
        # evaluate learner by the specified training and test datasets
        score = eval_classifier(learner, datapoints_train, labels_train, datapoints_test, labels_test, learner_params)
        # append evaluation scores to a list
        scores.append(score)
    # get average of scores and return
    return sum(scores) / k


###############################################################################
# For Auto Dataset
###############################################################################

path_auto_data = directory_data / 'small_auto_mpg.tsv'


def load_auto_data(path_data):
    '''
    Read a tsv file line by line and store informations of each line as a dictionary into a list.
        dictionaries whose keys are picked up from the first line of the file as fieldnames,
        and values are tab seprated values of each line.

    Params:
        path_data (file path) of auto data

    Returns:
        a list of information of each row of the file as a dictionary.
    '''

    numeric_fields = {
        'mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
        'acceleration', 'model_year', 'origin'
    }

    data = []
    # with file handler
    with open(path_data) as f_data:
        # create a DictReader iterator object on the file handler and loop over it line by line as datum
        for datum in csv.DictReader(f_data, delimiter='\t'):
            # csv.DirctReader object returns tab separated informations of each line as a dict (datum)
            # keys from the first line of the file, values from the current line
            # notice all values stored as strings

            # convert the numeric values from string to numbers
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])

            # append the converted datum into data
            data.append(datum)

    return data

# # test load_auto_data on small_auto_mpg.tsv
# print(load_auto_data(path_auto_data))


def stats(data, feature):
    '''
    Calculate the average and standard deviation of a feature in data.

    Params:
        data (a list of dicts): each dict represent an entry of the data,
            dict of features -> values
        feature (string): feature name or field name which is present in the data

    Returns:
        a tuple of (average, standard deviation)
    '''
    # pick up every values of the feature from data entries
    values = [entry[feature] for entry in data]
    # calculate the average of the values
    avg = sum(values) / len(values)
    # calculate the standard deviation of the values
    sqrdiff = [(value-avg)**2 for value in values]
    std = (sum(sqrdiff) / len(sqrdiff))**0.5
    # return the average and standard deviation as a tuple
    return avg, std


def standard(value, stats):
    '''
    Standardize a value (of a feature) based on the average
        and standard deviation of that feature.

    Params:
        value (int)
        stats (tuple): of average and standard deviation
    
    Returns:
        standardized value in a list
    '''
    avg, std = stats
    return [(value - avg) / std]


def one_hot(entry, entries):
    ''''
    Make a one-hot vector for entry given, based on entries.
        Example:
            entries = [11, 12, 13, 14]
            >>> one_hot(13, entries)
                [0, 0, 1, 0]
    
    Params:
        entry (int)
        entries (list of ints)

    Returns:
        a one-hot vector as a list
    '''
    vec = [0] * len(entries)
    vec[entries.index(entry)] = 1
    return vec


def raw(value):
    '''
    Returns the raw value in a list.
    '''
    return [value]


def auto_data_and_labels(auto_data_dict, featfuncs):
    '''
    Apply specified feature functions to (numeric) auto data
        and return data and labels as arrays.

    Params:
        auto_data (list of dicts): each dict is a data entry as field name -> value.
            See load_auto_data function.
        featfuncs (list of tuples): specified feature functions for field names to apply,
            tuples of (feature name (str), feature function (func)).
            Three option for feature functions:
                raw: use the original value
                standard: standardize the value
                one-hot: will one-hot encode the value
    Returns:
        a tuple of data and labels which feature functions applied
    '''
    # add the class (mpg) to the front of feature functions as raw
    featfuncs = [('mpg', raw)] + featfuncs

    # before applying feature functions to auto_data
    # we want to prepare stats and one-hot entries for features they need them
    
    # prepare stats for field names which are going to be standardized
    std = {feat: stats(auto_data_dict, feat) for feat, func in featfuncs if func == standard}
    # prepare entries for each one-hot field name
    entries = {feat: list(set([datum[feat] for datum in auto_data_dict])) \
                for feat, func in featfuncs if func == one_hot}
    # print('average and standard value for fields to be standardized:', std)
    # print('entries for fields to be one-hot:', entries)

    # applying feature functions to auto data and get an array out of it #
    
    # loop over datums in auto data
    lines = []
    for datum in auto_data_dict:
        # loop over feature functions to see what to extract from that datum
        line = []
        for feat, func in featfuncs:
            # for every feature name, we apply the function based on their feature function specified
            # and gather all the applied on a datum in a list we call line
            if func == standard:
                line.extend(func(datum[feat], std[feat]))
            elif func == one_hot:
                line.extend(func(datum[feat], entries[feat]))
            else:   # raw case
                line.extend(func(datum[feat]))
        # append the resultant line to the lines as an array
        lines.append(np.array(line))
    # make an array out of lines, which consists of data and labels
    data_labels = np.vstack(lines)
    # slice the data and labels from the data labels array
    return data_labels[:, 1:].T, data_labels[:, 0:1].T

# # test auto_data_and_labels for a specific feature functions
# featfuncs = [('displacement', standard),
#              ('acceleration', raw),
#              ('origin', one_hot)]
# auto_data = load_auto_data(path_auto_data)
# data, labels = auto_data_and_labels(auto_data, featfuncs)
# print('data:', data)
# print('labels:', labels)


###############################################################################
# For Review Dataset
###############################################################################

from string import punctuation, digits, printable
import re


path_review_data = directory_data / 'reviews.tsv'

sample_texts = [
    "There is a house in New Orleans.",
    "They call the9Rising Sun\n",
    "And it's been the ruin of many a ,poor boy",
    "And God, I know I'm one",
]

# stop words
path_stop_words = directory_data.parent / 'code_and_data_for_hw3' / 'stopwords.txt'
with open(path_stop_words) as f:
    STOP_WORDS = [word.strip() for word in f]


def load_review_data(path_data, maxline = -1):
    '''
    Read a tsv file line by line and store informations (sentiment and text)
        of each line as a dictionary into a list.

    Params:
        path_data (file path) of review data
        maxline (int, optional = -1): pick the first lines of the file to maxline number

    Returns:
        a list of information of rows of the file as dictionaries.
        each dictionary has keys as 'sentiment' and 'text',
        and values are int (-1 or +1) and str (review), respectively.
    '''
    basic_fields = {'sentiment', 'text'}
    # open file with file handler
    with open(path_data) as f_data:
        # loop over file line by line, by means of csv.DictReader
        data = []
        for nline, datum in enumerate(csv.DictReader(f_data, delimiter='\t')):
            # check the condition for reach to maxline
            if nline == maxline:
                break
            # delete unnecessary fields in each line
            for field in list(datum.keys()):
                if field not in basic_fields:
                    del datum[field]
            # convert sentiment to integer type
            if datum['sentiment']:
                datum['sentiment'] = int(datum['sentiment'])
            # append the line to data
            data.append(datum)
    # return data
    return data

# review_data = load_review_data(path_review_data)
# pprint(review_data[:5])


def extract_tokens(text, ngram = 1):
    '''
    Extract tokens out of a text by splitting text to sentences
        and filtering out digits and punctuations (except apostrophe),
        then consider the tokens as n-gram sequences of words.

        Sentences would be separated by dot, new line, question or exclamation marks.

    Params:
        text (string)
        ngram (int, optional = 1): value 1 associated with unigrams,
            value 2 as bigrams, and so on.
    
    Returns:
        a list of n-gram tokens as strings.
    '''
    # split text into sentences by (dot, new line, question or exclamation mark) characters
    sentences_str = re.split('\.|\n|\?|\!', text.lower())
    # filter empty strings after splitting
    sentences_str = list(filter(None, sentences_str))
    # clean the digits and punctuations (except apostrophe)
    sentences_splt = [re.findall(r"[A-Za-z']+", sentence) for sentence in sentences_str]
    # loop over sentences again and tokenize them by ngram length
    tokens_list = []
    for sentence in sentences_splt:
        tokens_list += [' '.join(sentence[start:start+ngram]) \
            for start in range(len(sentence) - ngram + 1)]
    return tokens_list


# sample_text = '\n'.join(sample_texts)
# print('-- text:\n', sample_text)
# print('-- extracted words:\n', extract_tokens(sample_text, ngram=2))


def bag_of_words(texts, filter_stopwords = False, ngram = 1):
    '''
    Create a bag-of-words dictionary from a list of string texts,
        by tokenizing them in n-gram sequences.

    Params:
        texts (list of strings): the corpus
        filter_stopwords (bool, optional = False): only can be applied in unigram case
        ngram (int, optional = 1): value 1 associated with unigrams,
            value 2 with bigrams, and so on.

    Returns:
        dictionary (token -> tokenid)
    '''
    dictionary = {}
    # loop over texts
    for text in texts:
        # extract tokens list from text
        tokens_list = extract_tokens(text, ngram)
        # loop over tokens list
        for token in tokens_list:
            # check the conditions and add the new tokens to dictionary
            # to have a tokenid for every tokens
            if token not in dictionary:
                # filter stopwords in the case of unigram
                if filter_stopwords and ngram == 1:
                    if token in STOP_WORDS: continue
                dictionary[token] = len(dictionary)
    # return dictionary
    return dictionary

# print('-- texts list:\n', sample_texts)
# dictionary = bag_of_words(sample_texts, filter_stopwords=False, ngram=2)
# print('-- bag-of-words:\n', dictionary)


def extract_bow_feature_vector(reviews, dictionary, count_tokens = False):
    '''
    Represent reviews data as a bag-of-words term-document feature matrix.

    Params:
        reviews (list of strings)
        dictionary (dict): bow of token -> tokenid
        count_tokens (bool, optional = False):
            if False, the representation is just about the existence,
                so the entries of the feature matrix would be ones and zeros.
            if True, the frequency of word appearance in the reviews will be taken into account.

    Returns:
        an array in the shape of (#tokens, #reviews)
        so each column is the bow representation of the corresponding review.
    '''
    # create zero array for term-document feature matrix to fill it later
    feature_matrix = np.zeros((len(dictionary), len(reviews)))
    # extract ngram number from the dictionary
    # by picking any key of the dictionary and count the number of its words
    ngram = len(list(dictionary.keys())[0].split())
    # loop over reviews
    for r, review in enumerate(reviews):
        # extract tokens out of the review
        review_tokens = extract_tokens(review, ngram)
        # loop over tokens
        for token in review_tokens:
            # if word is in dictionary,
            # change the corressponding position of term-document matrix by rule given
            if token in dictionary:
                if count_tokens: feature_matrix[dictionary[token], r] += 1
                else: feature_matrix[dictionary[token], r] = 1
    # return term-document matrix
    return feature_matrix

# print('-- text:\n', sample_texts)
# dictionary = bag_of_words(sample_texts, ngram=2)
# print('-- dictionary:\n', dictionary)
# sample_texts_bow_data = extract_bow_feature_vector(sample_texts, dictionary, count_tokens=False)
# print('-- bow representation of the text:\n', sample_texts_bow_data)


def reverse_dict(d):
    '''
    Return the reversed key, value of the dictionary d given.
    '''
    return {v: k for k, v in d.items()}


###############################################################################
# For MNIST Dataset
###############################################################################

from matplotlib.image import imread


mnist_directory = directory_data / 'mnist'


def load_mnist_single(path_data):
    '''
    Pick up (the first column of) handwritten digits of mnist image,
        each digit image as an array in the returned list of digit images.

    Params:
        path_data (file path): of the mnist png image

    Returns:
        a list of digit images, each as an 28 x 28 array with elements between 0 to 1
    '''
    # read the mnist image in path given by imread function
    img = imread(path_data)
    m, n = img.shape

    # each handwritten digit in mnist image is replaced into a 28 x 28 pixels window
    side_len = 28
    n_img = m // 28

    # loop over number of images
    # and pick up 28 x 28 arrays as digit images
    # notice that here we only pick the first row of the large mnist picture
    imgs = []
    for i in range(n_img):
        start_ind = i * side_len
        current_img = img[start_ind:start_ind+side_len, :side_len]
        imgs.append(current_img)

    # return the list of digit images extracted
    return imgs

# # test
# path_data = mnist_directory / 'mnist_train5.png'
# images_list = load_mnist_single(path_data)
# print(images_list[1])
# import matplotlib.pyplot as plt
# plt.imshow(images_list[1])
# plt.show()


def load_mnist_data(digits):
    '''
    Construct the dataset dictionary for digits given.
        Dataset setup in the form of:
            data[digit] = {
                'images': list of digit images as 28x28 arrays,
                'digits': a 1xlen(images) array with corresponding "digit" as its elements
            }

    Params:
        digits (list of integers in range(10)): digits we want to extract their images

    Returns:
        data (dict of digit -> {'images': ..., 'digits': ...})
    '''
    data = {}
    # loop over digits given
    for digit in digits:
        # load the list of digit images
        images = load_mnist_single(mnist_directory / f'mnist_train{digit}.png')
        # construct a list for the digits of images
        y = np.array([[digit] * len(images)])
        # place the images and their digits as a dictionary into data dict
        data[digit] = {'images': images, 'digits': y}
    
    return data

# # test
# data = load_mnist_data([1, 3])
# images3, digits3 = data[3]['images'], data[3]['digits']
# import matplotlib.pyplot as plt
# plt.imshow(images3[0])
# print(f'-- The shown image is digit {digits3[0]}.')
# plt.show()


def get_classification_accuracy(data, labels):
    '''
    10-fold cross-validation of perceptron with hyperparameter T set to 50
        on data and labels given.

    Params:
        data (dxn array)
        labels (1xn array)

    Returns:
        accuracy (float between 0 to 1)
    '''
    return xval_learning_alg(perceptron, data, labels, k=10, learner_params={'T': 50}, seed=0)