#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Functional Tools for Ridge Regression on Auto Data
==================================================

This module includes tools as:

    * Linear regression and its gradient:
        - linear regression and square loss
        - ridge regression
        - gradients in explicit (by hand calculation) and implicit (numerical) form
    
    * Stochastic gradient descent (sgd)
        - sgd and sgd test
        - ridge regression minimizer which wrap the needed functions for sgd and call it

    * Polynomial transformation tools (see hw3)

    * Evaluation
        by RMSE metric and cross-evaluation

    * Tools for work with auto data (see hw3)
        such as loading and making feature vectors out of them

    * Tools for work with textual data (see hw3)
        such as loading and making feature vectors out of them in bag-of-words representation
"""


import numpy as np
import matplotlib.pyplot as plt
import csv
import itertools, functools, operator


__author__ = "Saeed Entezari"
__email__ = "s.entezari144@gmail.com"



###############################################################
# Utilities
###############################################################


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



###############################################################
# Linear Regression and Its Gradient
###############################################################



# Linear regression and square loss
# ---------------------------------


def lin_reg(x, th, th0):
    '''
    Returns the (linear) prediction values of x datapoints,
        h(x; th, th0) = th.T @ x + th0,
        based on the predictor specified by th and th0 parameters.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        th (dx1 array)
        th0 (1x1 array or a scalar)

    Returns:
        a 1xn array of predictions
    '''
    return np.dot(th.T, x) + th0


def square_loss(x, y, th, th0):
    '''
    Calculate the square losses of every datapoints
        and put them in an array.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)

    Returns:
        a 1xn array of square losses.
    '''
    return (y - lin_reg(x, th, th0))**2


def mean_square_loss(x, y, th, th0):
    '''
    Get the mean square losses of all datapoints.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)
    
    Returns:
        a 1x1 array of mean square loss
    '''
    return np.mean(square_loss(x, y, th, th0), axis=1, keepdims=True)



# Gradient with respect to "theta"
# --------------------------------

def d_lin_reg_th(x, th, th0):
    '''
    Returns the gradients of linear regression for all datapoints in x
        with respect to theta.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        th (dx1 array)
        th0 (1x1 array or a scalar)

    Returns:
        a dxn array which each column is the gradient of
            the linear regression for corresponding datapoint in x
            (which is the same datapoint in x)
            with respect to theta.
    '''
    return x


def d_square_loss_th(x, y, th, th0):
    '''
    Calculate the gradients of every datapoints with respect to theta
        and put them in an array.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)

    Returns:
        a dxn array which each column is the gradient of
            the square loss for corresponding datapoint in x
            with respect to theta.
    '''
    return 2 * (lin_reg(x, th, th0) - y) * d_lin_reg_th(x, th, th0)


def d_mean_square_loss_th(x, y, th, th0):
    '''
    Get the mean square loss gradient with respect to theta.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)

    Returns:
        a dx1 array of mean square loss gradient with respect to theta.
    '''
    return np.mean(d_square_loss_th(x, y, th, th0), axis=1, keepdims=True)



# Gradient with respect to "theta0"
# ---------------------------------


def d_lin_reg_th0(x, th, th0):
    '''
    Returns the gradients of linear regression for all datapoints in x
        with respect to theta0.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        th (dx1 array)
        th0 (1x1 array or a scalar)

    Returns:
        a 1xn array which each column is the gradient of
            the linear regression for corresponding datapoint in x
            (which is 1)
            with respect to theta0.
    '''
    d, n = x.shape
    return np.ones((1, n))


def d_square_loss_th0(x, y, th, th0):
    '''
    Calculate the gradients of every datapoints with respect to theta0
        and put them in an array.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)

    Returns:
        a 1xn array which each column is the gradient of
            the square loss for corresponding datapoint in x
            with respect to theta0.
    '''
    return 2 * (lin_reg(x, th, th0) - y) * d_lin_reg_th0(x, th, th0)


def d_mean_square_loss_th0(x, y, th, th0):
    '''
    Get the mean square loss gradient with respect to theta0.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)

    Returns:
        a 1x1 array of mean square loss gradient with respect to theta0.
    '''
    return np.mean(d_square_loss_th0(x, y, th, th0), axis=1, keepdims=True)



# Ridge objective and its gradient
# --------------------------------


def ridge_obj(x, y, th, th0, lam):
    '''
    Returns the ridge objective of datapoints x with their targets y
        based on predictor with parameters theta and theta0
        and "regularization parameter lambda".

        Ridge objective is the same mean square loss (or empirical risk)
        added by regularization term lambda * norm of theta.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)
        lam (float): regularization parameter

    Returns:
        a 1x1 array of ridge objective value
    '''
    return mean_square_loss(x, y, th, th0) + lam * np.linalg.norm(th)**2


def d_ridge_obj_th(x, y, th, th0, lam):
    '''
    Calculate the gradient of ridge objective with respect to theta.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)
        lam (float): regularization parameter

    Returns:
        a dx1 array of ridge objective gradient with respect to theta.
    '''
    return d_mean_square_loss_th(x, y, th, th0) + 2 * lam * th


def d_ridge_obj_th0(x, y, th, th0, lam):
    '''
    Calculate the gradient of ridge objective with respect to theta0.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)
        lam (float): regularization parameter

    Returns:
        a 1x1 array of ridge objective gradient with respect to theta0.
    '''
    return d_mean_square_loss_th0(x, y, th, th0)


def ridge_obj_grad(x, y, th, th0, lam):
    '''
    Calculate the gradient of ridge objective with respect to theta and theta0.

    Params:
        x (dxn array): n d-dimensional datapoints arranged in columns
        y (1xn array): target values of datapoints in the same order of x
        th (dx1 array)
        th0 (1x1 array or a scalar)
        lam (float): regularization parameter

    Returns:
        a (d+1)x1 array of ridge objective gradient
            the first d elements for theta
            and the the last corresponds to theta0.
    '''
    grad_th = d_ridge_obj_th(x, y, th, th0, lam)
    grad_th0 = d_ridge_obj_th0(x, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])



# Numerical gradient calculator
# -----------------------------


def num_grad(f, delta=0.001):
    '''
    Wrapper function for numerical gradient calculator of function f,
        wrapped by the same function f, and delta as the telorance in approximating the gradient.

    f just takes a column array x as a whole parameter
        and returns a scalar.


    Test Case:
            create the numerical gradient calculator
            default delta is 0.001
        >>> df = num_grad(lambda x: np.sum(x))
            we want to calculate numerical gradient at point x
            notice the datatype of array x, suggested to be np.float64
        >>> x = np.array([[1.], [2.], [3.]])
            numerical gradient of function lambda x: np.sum(x) at point x
        >>> df(x)
        np.array([[1.], [1.], [1.]])
            as we expected by the hand calculation
    '''
    def df(x):
        d = len(x)
        grad = np.zeros((d, 1))
        for i in range(d):
            deltavec = np.zeros((d, 1))
            deltavec[i] = delta
            grad[i] = (f(x+deltavec) - f(x-deltavec)) / (2*delta)
        return grad
    return df



###############################################################
# Stochastic Gradient Descent
###############################################################



# Implement stochastic gradient descent
# -------------------------------------


def sgd(X, y, J, dJ, w0, step_size_fn, max_iter):
    '''
    Perform the stochastic gradient descent.

    Note that this function use a single parameter weight, not separated th and th0.
        So given X should be 1-augmented and initial weight parameter
        should have the same dimension of X datapoints.
        And so for functions J and dJ which should take datapoints in the same dimension.

    And again NOTE that 1-augmentation should be in the last coordinate position of datapoints.


    Params:
        X (dxn array): n d-dimensional datapoints arranged column-wise to train
        y (1xn array): corresponding target values
        J (func): cost function which calculates the square loss of a specified datapoint
            gets the inputs as:
                (dx1 array as a datapoint,
                1x1 array as its target,
                dx1 array as the current weight parameter)
            and returns a float scalar
        dJ (func): cost function gradient with respect to weight parameter.
            its input arguments is the same as J function,
            but returns a dx1 array of gradient
        w0 (dx1 array): the initial weight parameter
        step_size_fn (func): this function gets the step (iteration) number (zero-indexed)
            as input and returns a float step size for gradient descent as output
        max_iter (int): the number of iterations to perform

    Returns:
        the final weight parameter as a dx1 array,
        the list of random selected datapoints costs in each iteration as float,
        the list of weight parameters in the process as dx1 array vectors.
    '''
    d, n = X.shape
    fs, ws = [], []
    w = w0
    np.random.seed(0)
    for it in range(max_iter):
        # first we pick a random column out of datapoints array X
        # and record its cost and the current weight (parameter)
        j = np.random.randint(n)
        Xj, yj = X[:, j:j+1], y[:, j:j+1]
        fs.append(float(J(Xj, yj, w))); ws.append(w)

        # avoid the change of parameter in the last iteration
        if it == max_iter-1:
            return w, fs, ws

        # then we update the parameter based on the gradient of
        # the random picked datapoint and the current parameter
        w = w - step_size_fn(it) * dJ(Xj, yj, w)



def sgdTest():
    '''
    Test the stochastic gradient descent with a completely linear dataset
        (or another dataset which is provided),
        and print the results.
    '''
    def downwards_line():
        X = np.array([[0.0, 0.1, 0.2, 0.3, 0.42, 0.52, 0.72, 0.78, 0.84, 1.0],
                      [1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]])
        y = np.array([[0.4, 0.6, 1.2, 0.1, 0.22, -0.6, -1.5, -0.5, -0.5, 0.0]])
        # X = np.vstack([np.linspace(0, 1, 11), np.ones((1, 11))])
        # y = 20*X[:-1, :]
        print('X used\n', X)
        print('y used\n', y)
        return X, y
    

    # another dataset which is a complete linear dataset
    def complete_linear_dataset(slope=20):
        # interception of the line is set to 0
        X = np.vstack([np.linspace(0, 1, 11), np.ones((1, 11))])
        y = slope * X[:-1, :]
        print('X used\n', X)
        print('y used\n', y)
        return X, y


    # we want to give the ridge objective to the sgd as the J argument
    # ridge objective takes (seprated X, y, th, th0) format as its input
    # but formal parameter J of sgd takes (1-augmented X, y, w = stacked th and th0) format as input
    # so we need to construct a translator function which takes (1-augmented X, y, w) format as input
    # and returns ridge objective value by calling the ridge objective function with (seprated X, y, th, th0) format
    def J(Xj, yj, w):
        # translate ridge objective function from (1-augmented X, y, w) format to (separated X, y, th, th0) format
        return ridge_obj(Xj[:-1, :], yj, w[:-1, :], w[-1:, :], 0)


    # for constructing dJ, we want to use numerical gradient calculator
    # num_grad takes a function as input, a function whose takes weight parameter
    # and returns the objective value (or here, the cost value)
    def dJ(Xj, yj, w):
        def f(w): return J(Xj, yj, w)
        return num_grad(f)(w)


    def step_size_fn(i):
        return 1 / (i+1)**0.5


    # X, y = downwards_line()
    X, y = complete_linear_dataset(slope=20)
    d, n = X.shape
    w0 = np.zeros((d, 1))
    # now we are ready to call sgd function
    # before that, we want to set the seed to 0
    np.random.seed(0)
    w, fs, ws = sgd(X, y, J, dJ, w0, lambda i: 0.1, max_iter=1000)
    # print the results
    print('--- final weight\n', w)
    print('the first and the last element of fs\n', [fs[0], fs[-1]])
    print('the first and the last element of ws\n', [ws[0].tolist(), ws[-1].tolist()])
    print('length of fs', len(fs), 'and the length of ws', len(ws))


# # call the sgd test to see the results
# sgdTest()



# Ridge regression minimizer
# --------------------------


def ridge_min(X, y, lam, max_iter = 1000, cvg_plot = False, ax = None, nbin = None, pause = True):
    '''
    Returns th, th0 which minimize the ridge regression objective.

    NOTE: Assumes that X is NOT 1-extended. Interfaces to sgd function by
        1-extending and building corresponding initial weight parameter.

    Params:
        X (dxn array, separated format or not 1-extended): n d-dimensional datapoints arranged column-wise
            for train
        y (1xn array): their target values
        lam (float): regularization parameter (learning algorithm hyperparameter)
        max_iter (int, optional = 1000): the number of iterations sgd will perform (learning algorithm hyperparameter).
            NOTE: when we call xval_ridge_reg func, it calls rmse_eval_ridge_reg func multiple times
                and rmse_eval_ridge_reg calls this ridge_min func.
                insted of defining max_iter parameter in every functions in this series,
                if each of these function series get a kwargs and pass it to the next function,
                we can extract the max_iter variable in ridge_min to pass it into sgd
                and max_iter could be placed and adjusted from anywhere in the series of function calls.
        cvg_plot (bool, optional = False): draw the convergence plot
        ax (matplotlib axes, optional = None): axes which plot will be drawn on it.
            if not given, an axes will be created instead
        nbin (int, optional = None): if nbin given, convergence plot will be divided into nbin's
            where each bin will be averaged in x and y.
        pause (bool, optional = True): pause the plt GUI to see the plot

    Returns:
        th, th0 which is the final result of calling sgd function by
            step sizes obtained from svm_min_step_size_fn and
            number of iteration max_iter.
    '''
    d, n = X.shape
    # extend each datapoint coordinates by 1 to obtain 1-augmented X
    # the format which sgd works with
    X_extend = np.vstack([X, np.ones((1, n))])
    # set the initial weight parameter
    w0 = np.zeros((d+1, 1))

    def svm_min_step_size_fn(i):
        return 0.01 / (i+1)**0.5

    def J(Xj, yj, w):
        # takes the (1-augmented X, y, w) and returns the ridge objective value
        # notice that ridge objective we defined takes the (separated X, y, th, th0) input format
        return ridge_obj(Xj[:-1, :], yj, w[:-1, :], w[-1:, :], lam)

    def dJ(Xj, yj, w):
        # here we use the ridge objective gradient in the explicit form
        return ridge_obj_grad(Xj[:-1, :], yj, w[:-1, :], w[-1:, :], lam)

    # perform sgd
    np.random.seed(0)
    w, fs, ws = sgd(X_extend, y, J, dJ, w0, svm_min_step_size_fn, max_iter=max_iter)

    # convergence plot
    if cvg_plot:
        def mean(lst):
            return sum(lst) / len(lst)

        x, y = range(max_iter), fs
        if nbin:
            k = max_iter // nbin
            x = [mean(x[i*k:(i+1)*k]) for i in range(nbin)]
            y = [mean(y[i*k:(i+1)*k]) for i in range(nbin)]

        if ax is None: fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel('iteration number')
        ax.set_ylabel('objective value')
        if pause:
            plt.pause(1)
            input('\nConvergence plot is drawn.\nPress enter to close the plot.')

    # return th and th0 which minimize ridge objective
    return w[:-1, :], w[-1:, :]



###############################################################
# Polynomial Transformation
###############################################################


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



###############################################################
# Evaluation
###############################################################

def rmse_eval_ridge_reg(X_train, y_train, X_test, y_test, lam, max_iter):
    '''
    Evaluation the ridge regression predictor by training on train dataset
        then find the RMSE of the learned predictor on test dataset.

    Datapoints X are NOT 1-augmented.

    Params:
        X_train (dxn array)
        y_train (1xn array)
        X_test (dxm array)
        y_test (1xm array)
        lam (float): regularization parameter
        max_iter (int): number of sgd iterations

    Returns:
        RMSE of the learned predictor on test dataset as a float number
    '''
    th, th0 = ridge_min(X_train, y_train, lam, max_iter)
    return np.sqrt(mean_square_loss(X_test, y_test, th, th0))[0, 0]


def shuffle(X, y):
    '''
    Shuffle datapoints and their targets given.
        In fact, this function shuffle the inputs X and y column-wise in the same manner.

    The seed is set to 0.
    '''
    d, n = X.shape
    idx = list(range(n))
    np.random.seed(0)
    np.random.shuffle(idx)
    return X[:, idx], y[:, idx]


def xval_ridge_reg(X, y, lam, max_iter = 1000, k=10):
    '''
    Cross-validation evaluation of the ridge regression
        by performing sgd (through ridge minimizer function) and test with RMSE evaluation.
        Dataset will be divided into k chunks.

    Params:
        X (dxn array): the whole datapoints consisting of n d-dimensional datapoints,
            NOT 1-augmented!
        y (1xn array): their corresponding target values
        lam (float): regularization parameter
        max_iter (int, optional = 1000): number of sgd iterations
        k (int, optional = 10): dataset will be divided into k chunks to cross-validate
    
    Returns:
        cross-validation error which is the average of RMSEs, as a float number
    '''
    # shuffle dataset
    X, y = shuffle(X, y)

    # split dataset
    X_split = np.array_split(X, k, axis=1)
    y_split = np.array_split(y, k, axis=1)

    # cross-validation
    error_sum = 0
    for i in range(k):
        X_train = np.concatenate(X_split[:i] + X_split[i+1:], axis=1)
        y_train = np.concatenate(y_split[:i] + y_split[i+1:], axis=1)
        X_test, y_test = X_split[i], y_split[i]
        # train and test
        error_sum += rmse_eval_ridge_reg(X_train, y_train, X_test, y_test, lam, max_iter)

    return error_sum / k



###############################################################
# For Auto Dataset
###############################################################

# for more information you can also see hw3 file, part 2
# which is about loading and representing auto data


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
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)

    return data


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
    values = [entry[feature] for entry in data]
    avg = sum(values) / len(values)
    sqrdiff = [(value-avg)**2 for value in values]
    std = (sum(sqrdiff) / len(sqrdiff))**0.5
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


def auto_data_and_targets(auto_data_dict, featfuncs):
    '''
    Apply specified feature functions to (numeric) auto data
        and return data and their targets as two arrays.

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
        a tuple of data and their targets which feature functions applied on them
    '''
    featfuncs = [('mpg', raw)] + featfuncs
    
    std = {feat: stats(auto_data_dict, feat) for feat, func in featfuncs if func == standard}
    entries = {feat: list(set([datum[feat] for datum in auto_data_dict])) \
                for feat, func in featfuncs if func == one_hot}

    lines = []
    for datum in auto_data_dict:
        line = []
        for feat, func in featfuncs:
            if func == standard:
                line.extend(func(datum[feat], std[feat]))
            elif func == one_hot:
                line.extend(func(datum[feat], entries[feat]))
            else:
                line.extend(func(datum[feat]))
        lines.append(np.array(line))

    data_targets = np.vstack(lines)
    return data_targets[:, 1:].T, data_targets[:, 0:1].T


def std_y(row_array):
    '''
    Standardize row array (target values).
    
    Params:
        row_array (1xn array): target values

    Returns:
        standardized row array (1xn array),
        average value (float),
        standard deviation (float)
    '''
    mu = np.mean(row_array, axis=1)
    sigma = np.sqrt(np.mean((row_array - mu)**2, axis=1))
    return (row_array - mu) / sigma, mu[0], sigma[0]



###############################################################
# For Textual Feature
###############################################################


def load_text_data(path_data, maxline = -1):
    '''
    Read a tsv file (specifically auto data) line by line and store informations (car_name)
        of each line as a dictionary into a list.

    Params:
        path_data (file path) of auto data
        maxline (int, optional = -1): pick the first lines of the file to maxline number

    Returns:
        a list of information of car_name's as dictionaries.
        each dictionary has a key as 'car_name',
        and its value is str (car_name).
    '''
    basic_fields = {'car_name'}
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
            # append the line to data
            data.append(datum)
    # return data
    return data

# path_data = 'auto-mpg-regression.tsv'
# data_dict = load_text_data(path_data)
# print('\nfirst 3 line of auto data loaded as:\n', data_dict[0:3])


def text_data_dict_to_list(data_dict):
    '''
    Convert the list of datum dictionaries extracted from auto data car_name field
        into a list of that datum value which is car_name value in string.
    '''
    return [datum['car_name'] for datum in data_dict]

# data_texts = text_data_dict_to_list(data_dict)
# print('\nfirst 3 car names:', data_texts[0:3])


def extract_tokens(text):
    '''
    Extract tokens out of a text by splitting it.

    Params:
        text (string)

    Returns:
        a list of splitted words in the text as strings.
    '''
    return text.split()


def bag_of_words(texts):
    '''
    Create a bag-of-words dictionary from a list of string texts.

    Params:
        texts (list of strings): the corpus

    Returns:
        dictionary (token -> tokenid)
    '''
    dictionary = {}
    # loop over texts
    for text in texts:
        # extract tokens list from text
        tokens_list = extract_tokens(text)
        # loop over tokens list
        for token in tokens_list:
            # check the conditions and add the new tokens to dictionary
            # to have a tokenid for every tokens
            if token not in dictionary:
                dictionary[token] = len(dictionary)
    # return dictionary
    return dictionary

# dictionary = bag_of_words(data_texts)
# print('\nfirst 8 (token, tokenid) of dictionary:', sorted(dictionary.items(), key=lambda x: x[-1])[:8])


def extract_bow_feature_vector(texts, dictionary):
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
    feature_matrix = np.zeros((len(dictionary), len(texts)))
    # loop over texts
    for t, text in enumerate(texts):
        # extract tokens out of the texts
        text_tokens = extract_tokens(text)
        # loop over tokens
        for token in text_tokens:
            # if word is in dictionary,
            # change the corressponding position of term-document matrix by rule given
            if token in dictionary:
                feature_matrix[dictionary[token], t] = 1
    # return term-document matrix
    return feature_matrix

# data_array = extract_bow_feature_vector(data_texts, dictionary)
# print('\nfirst 3 car names are encoded as (sliced and transposed, so each datapoint is shown in a row):\n:',
#     data_array[:,:3].T)