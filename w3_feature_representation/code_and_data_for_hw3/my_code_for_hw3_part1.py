#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Polynomial transformation: implementation and test
--------------------------------------------------

Tools of making polynomial transformation function which transforms raw data into polynomial basis.

Test perceptron with a linear classifier in the transformed space and see the non-linear separator in
the prior (untransformed) space (2D to have a visualization).

Plotting functions and some simple datasets are included.
"""

import itertools
import functools
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


__author__ = "Saeed Entezari"
__email__ = "s.entezari144@gmail.com"


################################################################
# Plotting
################################################################


def tidy_plot(xmin, xmax, ymin, ymax, ax = None, center = False, equal_aspect = True, grid = True,
                title = None, xlabel = None, ylabel = None):
    '''
    Set up axes for 2D plotting.

    Parameters:
        xmin, xmax, ymin, ymax (float): plot extents
        ax (matplotlib axes, optional = None): if axes is given, make it tidy.
            Otherwise, make a tidy axes.
        center (bool, optional = False): if True, just zero axes would be ticked and shown.
            if False, axes would be shown on the borders,
            but zero axes would be shown without ticks.
        title, xlabel, ylabel (string, optional = None)
        grid (bool, optional = True): draw a grid for plot
        equal_aspect (bool, optional = True): equal unit length for axes

    Returns: ax (matplotlib.axes.Axes instance)
    '''
    # plt.ion()
    if ax is None: fig, ax = plt.subplots()

    epsx = 0.1 * (xmax-xmin)
    epsy = 0.1 * (ymax-ymin)
    ax.set_xlim(xmin-epsx, xmax+epsx)
    ax.set_ylim(ymin-epsy, ymax+epsy)

    if center:
        ax.spines[['left', 'bottom']].set_position('zero')
        ax.spines[['right', 'top']].set_visible(False)
    else:
        ax.spines[['right', 'top']].set_visible(False)
        ax.plot([0, 0], [ymin-epsy, ymax+epsy], c='k', lw=0.5)
        ax.plot([xmin-epsx, xmax+epsx], [0, 0], c='k', lw=0.5)

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if equal_aspect: ax.set_aspect('equal')
    if grid: ax.grid(visible=True, ls=':')

    return ax


def plot_data(datapoints, labels, ax = None, xmin = None, xmax = None, ymin = None, ymax = None,
                clear = False, **kwargs):
    '''
    Scatter plot data in 2D with labels as colors.

    Parameters:
        datapoints (2xn array): n 2d datapoints
        labels (1xn array): entries from set {-1, +1}
        ax (matplotlib axes, optional = None): if given, draw on given axes.
            if not given, draw a new axes based on datapoints.
        xmin, xmax, ymin, ymax (float, optional = None): set limits of the plot.
            if not given, would be set based on datapoints.
        clear (bool, optional = False): in case of axes is given,
            it could be True to clear the axes before plotting.

    Returns:
        ax (matplotlib axes)
    '''
    # get marker value from kwargs
    marker = kwargs.get('marker', '.')
    # if axis is not given, make a new axis
    if ax is None:
        # if limits not given, take limits from datapoints
        if xmin == None: xmin = np.min(datapoints[0,:])
        if xmax == None: xmax = np.max(datapoints[0,:])
        if ymin == None: ymin = np.min(datapoints[1,:])
        if ymax == None: ymax = np.max(datapoints[1,:])
        # in case of max min equality, for example if we have just a datapoint
        # then extent the limits
        if xmax == xmin:
            xmin = xmin - 1
            xmax = xmax + 1
        if ymax == ymin:
            ymin = ymin - 1
            ymax = ymax + 1
        # create axes by the help of tidy_plot()
        ax = tidy_plot(xmin, xmax, ymin, ymax)
        # save limits into xlim and ylim variables for compatibility
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # if axis is given but if clear is True
    elif clear:
        # keep track of limits then clear plot
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        # clear the ax then make it tidy with limits we have
        ax.clear()
        ax = tidy_plot(xlim[0], xlim[1], ylim[0], ylim[1], ax=ax)
    # otherwise, just take limits of plot before plotting
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # transform labels to colors for plotting
    colors = np.choose(labels > 0, ['r', 'g'])[0]
    # scatter plot
    ax.scatter(datapoints[0,:], datapoints[1,:], c=colors, marker=marker)
    # set limits again to try to keep them from moving around
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # return axes
    return ax


def plot_nonlinear_separator(predictor, ax = None, npix = 30,
                                xmin = None, xmax = None, ymin = None, ymax = None):
    '''
    Plot non-linear separator in 2D, by a predictor given.
        Prediction value would be calculated for every pixel of the axes
        and it would be painted accordingly.
        Notice that you must specify an axisting axes or/and limits.

    Params:
        predictor (func): a function which gets two coordinates x and y
            as two float arguments respectively, and returns the value of prediction
            as int between -1 and +1.
        ax (matplotlib axes, optional = None)
        npix (int, optional = 30): number of pixels in each dimension.
        xmin, xmax, ymin, ymax (int, optional = None)

    Returns:
        None
        There is no need to return anything,
        because image would be displayed on the axes
    '''
    # plot limits
    if ax == None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    # create Colormap and Normalize instances
    cmap = colors.ListedColormap(['silver', 'white'])
    norm = colors.BoundaryNorm([-2, 0, 2], cmap.N)

    # create the image data array by giving every points to the predictor
    ima = np.array([[predictor(x, y) \
                        for x in np.linspace(xmin, xmax, npix)] \
                            for y in np.linspace(ymin, ymax, npix)])

    # create the image on the axes by imshow()
    im = ax.imshow(np.flipud(ima), cmap=cmap, norm=norm,
                    extent=[xmin, xmax, ymin, ymax], interpolation='none')
    # there is no need to return, image is displayed on the axes

################################################################
# Utilities
################################################################

def rv(value_list):
    '''
    Returns row vector (1xd array) from a list of values.
    '''
    return np.array([value_list])

def cv(value_list):
    '''
    Returns column vector (dx1 array) from a list of values.
    '''
    return np.transpose(rv(value_list))

def prediction(x, th, th_0):
    '''
    Prediction of label of datapoints x (dxn array)
        with respect to separator specified by parameters
        theta (dx1 array) and theta_0 (1x1 array).
    
    Returns (1xn array of) sign (int in {-1, 0, 1})
    '''
    return np.sign(np.dot(th.T, x) + th_0)

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

################################################################
# Data Sets
################################################################

def super_simple_separable_through_origin():
    '''
    Returns 2x4 (dxn) data array and their labels (1x4).
    '''
    data = np.array([[2, 3, 9, 12],
                     [5, 1, 6, 5]])
    labels = np.array([[1, -1, 1, -1]])
    return data, labels

def super_simple_separable():
    '''
    Returns 2x4 (dxn) data array and their labels (1x4).
        datapoints are seprable but not through origin.
    '''
    data = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    labels = np.array([[1, -1, 1, -1]])
    return data, labels

def xor():
    '''
    Returns 2x4 (dxn) data array and their labels (1x4).
        each label is the xor (result in {-1, +1}) of its datapoint's coordinates.
    Datapoints are not seprable.
    '''
    data = np.array([[1, 2, 1, 2],
                     [1, 2, 2, 1]])
    labels = np.array([[1, 1, -1, -1]])
    return data, labels

def xor_centered():
    '''
    Returns 2x4 (dxn) data array and their labels (1x4).
        each label is the xor (result in {-1, +1}) of its datapoint's coordinates.
        Datapoints are centered around origin.
    Datapoints are not seprable.
    '''
    data = np.array([[-1, 1, -1, 1],
                     [-1, -1, 1, 1]])
    labels = np.array([[-1, 1, 1, -1]])
    return data, labels

def xor_more():
    '''
    Returns 2x8 (dxn) data array and their labels (1x8).
        4 labels are the xor (result in {-1, +1}) of their datapoints's coordinates.
        and other 4 extra datapoint with their labels.
    Datapoints are not seprable.
    '''
    data = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    labels = np.array([[1, 1, -1, -1, 1, 1, -1, -1]])
    return data, labels

################################################################
# Polynomial Transformation
################################################################

# we want to create a polynomial transofrmation function
# which gets raw feature vectors as input
# and returns transformed feature vectores as output
# to do this we have 2 ways:

# (1)
# we can import polynomial_transformation module written by me and use make_poly_trans_func
# not recommended for big data

# (2)
# use Functional Programming Modules in python, like:
### itertools.combinations_with_replacement(iterable, k)
#       Gives a generator which
#       Returns k-length (combination) tuples of elements from the input iterable (set to be choosen from),
#       allowing individual elements to be repeated more than once.
#           >>> for combTuple in itertools.combinations_with_replacement('ABC', 2):
#           ...     print(combTuple)
                #  ('A', 'A')
                #  ('A', 'B')
                #  ('A', 'C')
                #  ('B', 'B')
                #  ('B', 'C')
                #  ('C', 'C')
#           >>> for combTuple in itertools.combinations_with_replacement(range(2), 3):
#           ...     print(combTuple)
                # indicate the indices of polynomial transformation of order 3, on raw feature vector [x0, x1]
                #  (0, 0, 0) -> x0 * x0 * x0 = x0^3 x1^0 ~ equivalent with powersTuple (3, 0);
                #                find out the meaning of powersTuple in polynomial_transofrmation.py
                #  (0, 0, 1) -> x0 * x0 * x1 = x0^2 x1^1 ~ (2, 1)
                #  (0, 1, 1) -> x0 * x1 * x1 = x0^1 x1^2 ~ (1, 2)
                #  (1, 1, 1) -> x1 * x1 * x1 = x0^0 x1^3 ~ (0, 3)

### functools.reduce(function, iterable[, initializer])
#       Apply function of two arguments cumulatively to the items of iterable,
#       from left to right, so as to reduce the iterable to a single value.
#       If optional initializer is present, it is placed before the items of the iterable
#       and serve as a default when the iterable is empty.
#       If initializer is not given and iterable contains only one item, return the first item,
#                                                contains no item, interrupt and raise a TypeError.
#           >>> functools.reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])
                # ((((1+2)+3)+4)+5) = 15
#           >>> functools.reduce(lambda x, y: x+y, [1, 2, 3, 4, 5], 100)
                # (((((100+1)+2)+3)+4)+5) = 115
#           >>> functools.reduce(lambda x, y: x+y, [], 100)
                # 100
#           >>> functools.reduce(lambda x, y: x+y, [])
                # TypeError: reduce() of empty sequence with no initial value
#           >>> functools.reduce(lambda x, y: x+y, [1])
                # 1
#           >>> functools.reduce(lambda x, y: x*y, range(1, 4+1))
                # 24 = 4!

### operator.mul(a, b)
#       Returns a * b, for a and b numbers.
#       This is the same as multiplication.


################################################################
# Way (2): Neat code by functional programming modules of python

def mul(seq):
    '''
    Multiplies every elements of a sequence.
        If sequence is empty, returns 1.

    Params:
        seq (list of numbers)

    Returns:
        multiplied value (number)
    '''
    return functools.reduce(operator.mul, seq, 1)


def make_poly_trans_func(K):
    '''
    Wrap polynomial transformation function for max order K given.
    '''
    def poly_transformer(raw_features):
        '''
        Transforms raw feature vectors consisting of n d-dimensional raw feature vector
            in polynomial basis to order K.

        Params:
            raw_features (dxn array)

        Returns:
            transformed feature vectors (Sxn array)
                S is the sum of the number of polynomial features to max order K.
        '''
        d, n = raw_features.shape
        # loop over raw datapoints
        result = []
        for j in range(n):
            # loop over order k, from 0 to K
            transfeat = []
            for k in range(K+1):
                # get the combinations_with_replacement of indices,
                # for indices in range(d) and combinations of order k
                indexTuples = itertools.combinations_with_replacement(range(d), k)
                # loop over combinations of order k
                for it in indexTuples:
                    # make polynomial feature for each combination
                    # by multiplying the value of corresponding indices in raw datapoint
                    # then append it to the polynomial feature vector
                    transfeat.append(mul([raw_features[i, j] for i in it]))
            # append the transformed feature vectore for this raw datapoint to the result list (of arrays)
            result.append(cv(transfeat))
        # make an array out of the result and return the transformed input
        return np.hstack(result)

    return poly_transformer

# # test case
# K = 2
# raw_features = np.array([[2, 4], [3, 5]])
# poly_trans_func = make_poly_trans_func(K)
# print('Raw features as input:\n', raw_features)
# poly_features = poly_trans_func(raw_features)
# print(f'Transformed features in polynomial basis to order K = {K}:\n', poly_features)

################################################################
# Learner: Perceptron
################################################################

def perceptron(datapoints, labels, params = {}, hook = None, count_mistakes = False):
    '''
    Perform perceptron algorithm for datapoints with labels,
        by going through dataset T times.

    Parameters:
        datapoints (dxn array)
        labels (1xn array): values in {-1, 0, +1}
        params (dict, optional = {}): extra parameters to algorithm,
            such as T, the number of iterations through dataset.
        hook (func, optional = None): a function for plotting separator, if given.
        count_mistakes (bool, optional = False): if True, prints the number of mistake.

    Returns:
        final th (dx1 array)
        final th_0 (1x1 array)
    '''
    # if T not in params, set it to 100
    T = params.get('T', 100)
    # initialization
    d, n = datapoints.shape
    th = np.zeros((d, 1))
    th_0 = np.zeros((1, 1))
    # iteration
    nmistakes = 0
    for iter in range(T):
        for i in range(n):
            y_i = labels[0, i]
            x_i = datapoints[:, i:i+1]
            if y_i * prediction(x_i, th, th_0) <= 0:   # mistake and update
                # mistake counter
                nmistakes += 1
                # updata parameters
                th = th + y_i * x_i
                th_0 = th_0 + y_i
                # hook
                if hook:
                    hook((th, th_0))

    if count_mistakes: print('Number of misatkes happened:', nmistakes)
    return th, th_0

################################################################
# Tests
################################################################

def test_linear_classifier_in_trans_space(datagen, transfunc, learner, learner_params = {},
                                            draw = True, pause = True, count_mistakes = False):
    '''
    Test linear classifier in transformed space in action.
        In fact, this function first transforms data generated by data generator
        to get the transformed data in transformed space,
        and then perform (call the) learner which has linear hypothesis in the transformed feature space.
        If you want to draw, the non-linear separator in raw space
        (which is the reverse transformation of linear separator in transformed space)
        would be drawn in each step of the learning.

        Raw feature space should be 2-dimensional.

    Params:
        datagen (func): data generator function with all default arguments will be called
        transfunc (func): transformation function, which gives raw data and returns transformed data
        learner (func): perceptron or averaged perceptron
        draw (bool, optional = True): if True, the hook function would be made for
            2D plotting each step (mistakes) of learning algorithm
        pause (bool, optional = True): if True, for going to next step of learning algorithm,
            user should press the enter key
        count_mistakes (bool, optional = False): print the number of mistakes happened at the end

    Returns:
        None
        Print learned parameters and score on them.
        This function just made the hook function for plotting (if draw)
        and prepare learner arguments then call it.
    '''
    # get the raw dataset from data generator
    raw_data, labels = datagen()
    # get the transformed data by giving raw data to the transformer
    trans_data = transfunc(raw_data)

    # if draw, define the axes and hook function for plotting
    if draw:
        ax = plot_data(raw_data, labels)
        def hook(params):
            th, th_0 = params
            print('th =', th.T, 'th_0 =', th_0)
            plot_nonlinear_separator(
                lambda x, y: int(prediction(transfunc(cv([x, y])), th, th_0)), # predictor
                ax=ax
            )
            plt.pause(0.05)
            if pause: input('Continue?')
    else:
        hook = None

    # pass the prepared arguments to learner
    learned_th, learned_th_0 = learner(trans_data, labels, hook=hook,
                                        params=learner_params, count_mistakes=count_mistakes)
    # final result
    print('Learner finished!')
    if hook: hook((learned_th, learned_th_0)); input('Final plot is shown, Continue?')
    print('Final result:\nth =', learned_th, 'th_0 =', learned_th_0)
    print('Score =', score(trans_data, labels, learned_th, learned_th_0))

# # test case
# K = 3
# poly_trans = make_poly_trans_func(K)
# test_linear_classifier_in_trans_space(
#     xor, poly_trans, perceptron, learner_params={'T':100},
#     draw=True, pause=False, count_mistakes=True
# )