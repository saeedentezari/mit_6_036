#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Ridge Regression on Auto Data
=============================


In this file, we try to make feature vectors out of auto data by various feature choices
    and transform the obtained feature vectors in polynomial basis of orders 1, 2, or 3.
    Also we train and evaluate ridge regression on these datasets with a range of regularization parameters.

In the last section of this homework, we add the textual data in our feature vectors to see whether it's helpful.

Run the file and go through each section step by step.
"""


import my_code_for_hw5 as hw5
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


__author__ = "Saeed Entezari"
__email__ = "s.entezari144@gmail.com"


directory = Path('~/mit_6_036/w5_regression')



print('\n---------------------------------------- Prerequisite ----------------------------------------')

# load auto-mpg-regression.tsv file which is in the same directory as this file
path_auto_data = directory / 'auto-mpg-regression.tsv'
auto_data_dict = hw5.load_auto_data(path_auto_data)

# now we need feature sets to make a feature array out of auto_data_dict format
# notice that we indexed feature choice as 0 and 1, unlike the hw5 original text which is set to 1 and 2
featfuncs0 = [('cylinders', hw5.standard),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]
featfuncs1 = [('cylinders', hw5.one_hot),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot)]
print('\nbefore going through the exercise, we want to introduce the feature choices as follows:')
print('\nfeature set 0:', [(feature_name, feature_func.__name__) for feature_name, feature_func in featfuncs0])
print('\nfeature set 1:', [(feature_name, feature_func.__name__) for feature_name, feature_func in featfuncs1])

# construct the standard format dataset arrays (feature vectors)
auto_datapts = [0, 0]   # save the datapoints of different feature choices into a list
auto_datapts[0], auto_targets = hw5.auto_data_and_targets(auto_data_dict, featfuncs0)
auto_datapts[1], _ = hw5.auto_data_and_targets(auto_data_dict, featfuncs1)  # auto targets is the same for the two feature choices above

# input datapoints which need standardization are standardized by applying feature functions on them
# now it's time to standardize auto targets
auto_targets, trg_mu, trg_sigma = hw5.std_y(auto_targets)
# since the targets are standardized, the errors and evaluations which uses these standardized target
# are standardized too, so it should be reported by multiplying in target standard deviation to get the erros in mpg unit

# print('first datapoint featured by featfuncs0\n', auto_datapts[0][:,0:1])
# print('first standardized datapoint target\n', auto_targets[:, 0:1], '\nby mu and sigma\n', trg_mu, trg_sigma)



print('\n------------------------------------- One trial combination -------------------------------------')


K = 1
lam = 0

print(f"""\nwe consider:
    feature choice: featfuncs0
    polynomial transformation order: {K}
    regularization parameter: {lam}
    as a trial combination to train ridge regression model and evaluate it by 10-fold cross-validation on this dataset.""")


poly_transformer = hw5.make_poly_trans_func(K)
auto_datapts_trans = poly_transformer(auto_datapts[0])
print('\nshape of auto datapoints featured by featfuncs0 (without polynomial transformation):', auto_datapts[0].shape)
print(f'shape of auto datapoint featured by featfuncs0 transformed in poly order {K}:', auto_datapts_trans.shape)


# to give the transformed datapoints to xval_ridge_reg or equivalently ridge_min function,
# we should notice that ridge_min takes NOT 1-augmented datapoints
# and itself extend the datapoints array by 1 at the last coordinate for giving to sgd
# which the last 1-augmented coordinate plays the associated role of th0 in sgd function

# so we should delete polynomial 1-augmentation from the first coordinate of the transformed datapoints
def del_poly_augment(X):
    '''
    Delete the first coordinates of polynomial transformed X which are 1-augmented.
    '''
    return X[1:]

auto_datapts_trans = del_poly_augment(auto_datapts_trans)
print('\npolynomial 1-augmentation has been deleted and the new shape of transformed auto datapoints are:',
    auto_datapts_trans.shape)

# now auto dataset is ready to evaluation with the regularization parameter specified
std_error = hw5.xval_ridge_reg(auto_datapts_trans, auto_targets, lam)
# the error of standardized targets should be multiplied by sigma to get the real unit error
print(f'\ncross-validation error of ridge regression with lam {lam} on transformed auto datapoints and standardized targets is:',
    round(std_error, 3))
print(f'error of real unit targets with sigma {round(trg_sigma, 3)}:', round(std_error * trg_sigma, 3))

input('\ncontinue?')



# ---------------------------------------- utility functions ----------------------------------------



# evaluation for combinations
# ---------------------------


def del_poly_augment(X):
    '''
    Delete the first coordinates of polynomial transformed X which are 1-augmented.
    '''
    return X[1:]


def add_to_dict(d, value, *keys):
    '''
    Add the value to dictionary d in the path given by args input
        in the format of nested dictionaries.

    For example:
        >>> d = {}
        >>> d = add_to_dict(d, -1, 'a', 1, 2)
        >>> d['a'][1][2]
            -1
        >>> d
            {'a': {1: {2: -1}}}
        >>> d = add_to_dict(d, -10, 'b', 1, 3)
        >>> d
            {'a': {1: {2: -1}}, 'b': {1: {3: -10}}}
        >>> d['b'][1][3]
            -10

    Params:
        d (dict) to add
        value: any python object
        *keys (non-keyword arguments): path chain to add (and access) the value.
            All of the *keys inputs should be immutable objects.

    Returns:
        the added dictionary
    '''
    # go to the last layer #
    d_nest = d
    for key in keys[:-1]:
        if key not in d_nest:
            # create the key in this layer with an empty dict value
            d_nest[key] = {}
        # move one layer deeper
        d_nest = d_nest[key]

    # add the key value in the last layer #
    d_nest[keys[-1]] = value

    # notice that changes occured in-place, because of dict mutability
    return d


def rmse_all_combinations(auto_datapts, auto_targets, trg_sigma):
    '''
    Cross-evaluate ridge regression on auto dataset by RMSE metric.

    Combinations are:
        * featuring datapoints by feature choices number [0 and 1]
            (featfuncs0 and featfuncs1 defined at the beginning of this file)
        * transforming featured datapoints in polynomial basis of order [1 and 2 and 3]
        * cross-evaluating ridge regression by regularization parameters
            [0.00, 0.01, ..., 0.10] for K = 1 or 2, and
            [0, 20, ..., 200] for K = 3

            you can generalize this function by passing combinations into it.
    

    Params:
        auto_datapts (list of Dxn arrays): auto_datapts[i] is the auto datapoints featured by feature choice i.
            At the beginning of this file we made ready two feature datapoints by two feature choices we had.
        auto_targets (1xn array): corresponding targets which are standardized.
        trg_sigma (float): standard deviation of the targets

    Returns:
        a nested dictionary (dictionaries into dictionaries) of errors (in mpg units) for all combinations,
        for example the error of
            featfuncs_num = 0
            K = 1
            lam = 0.01
        can be accessed by error_dict[0][1][0.01]
    '''
    def rmse_one_combination(featfuncs_num, K, lam):
        '''
        Cross-evaluate ridge regression on auto dataset for a single combination given,
            and returns RMSE in mpg units as a float number.
        '''
        poly_transformer = hw5.make_poly_trans_func(K)
        auto_datapts_trans = poly_transformer(auto_datapts[featfuncs_num])
        auto_datapts_trans = del_poly_augment(auto_datapts_trans)
        # we can save transformed auto datapoints into a dictionary 
        # to avoid multiple transformation for different lambdas
        std_error = hw5.xval_ridge_reg(auto_datapts_trans, auto_targets, lam)
        return std_error * trg_sigma

    err_dict = {}
    for featfuncs_num in [0, 1]:
        for K in [1, 2, 3]:
            lam_list = np.arange(0, 220, 20) if K == 3 else np.arange(0, 0.11, 0.01)
            for lam in lam_list:
                error = rmse_one_combination(featfuncs_num, K, lam)
                err_dict = add_to_dict(err_dict, error, featfuncs_num, K, lam)
    
    return err_dict



# handy read and write functions
# ------------------------------

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



# minimum finder
# --------------


def min_finder(err_dict, featfuncs_nls, Kls):
    '''
    Find the minimum error in err_dict and its corresponding combination.

    lambda regularization parameters to investigate is set to:
        np.arange(0, 220, 20) if K == 3
        np.arange(0, 0.11, 0.01) if K == 1 or 2

    Params:
        err_dict (dict): nested dictionary of
            err_dict[featfuncs_num][K][lam] = error value for combination
        featfuncs_nls (list): of featfuncs numbers we want to find the minimum in them
        Kls (list): of K numbers we want to find the minimum in them

    Returns:
        min_error (float): minimum error found in search scope
        min_comb (tuple): of
            featfuncs_num,
            K,
            lam
    '''
    # set the first error value as minimum
    min_err = err_dict[featfuncs_nls[0]][Kls[0]][0]
    min_comb = (0, 0, 0)
    for featfuncs_num in featfuncs_nls:
        for K in Kls:
            lam_list = np.arange(0, 220, 20) if K == 3 else np.arange(0, 0.11, 0.01)
            for lam in lam_list:
                if err_dict[featfuncs_num][K][lam] < min_err:
                    min_err = err_dict[featfuncs_num][K][lam]
                    min_comb = (featfuncs_num, K, lam)
    
    return min_err, min_comb



print('\n------------------------------- make ready the error dict into a file -------------------------------')


made_ready = input("""\nhave you made ready the error dictionary file earlier?
    if not so, enter [n] as input to make it ready now.
    otherwise just enter and pass, to read the prepared file by file name 'err_dict.pk'.
    
    enter your input:""")
if made_ready == 'n':
    print(f"""\nerror dictionary for all combinations is computing and will be writed on file
    {directory / 'err_dict.pk'} for further uses.""")
    err_dict = rmse_all_combinations(auto_datapts, auto_targets, trg_sigma)
    write_on_file(err_dict, directory / 'err_dict.pk')
else:
    err_dict = load_from_file(directory / 'err_dict.pk')
input('\ncontinue to next section?')



print('\n----------------------------------- Minimum erros and combinations -----------------------------------')


print("""\nfirst of all we want to find the minimum error (in mpg) and its corresponding combination of
    (feature choice number, polynomial transformation order K, regularization parameter lambda)
    across the whole combinations we considered in this homework until now, which is
        featfuncs: [0, 1]
        poly. trans. order: [1, 2, 3]
        reg. param. lambda: [0.01, 0.02, ..., 0.1] for K = 1 or 2 | [0, 20, ..., 200] for K = 3""")

featfuncs_nls, Kls = [0, 1], [1, 2, 3]
min_error_all, min_comb_all = min_finder(err_dict, featfuncs_nls, Kls)
print('\nminimum error found:', round(min_error_all, 3))
print('for the combination:', min_comb_all)
print("""so the best result is for:
    featfuncs1, which cylinders and origin is one-hot encoded, and polynomial order 2 withoud regularization.""")

print("""\nnow, what if we want to fit an order 3 polynomial model with the first feature set?
    what would be the regularization parameter?""")
featfuncs_nls, Kls = [0], [3]
min_error_K3, min_comb_K3 = min_finder(err_dict, featfuncs_nls, Kls)
print('the best regularization parameter found:', min_comb_K3[-1])
print('with the error:', round(min_error_K3, 3))


input('\ncontinue?')




print('\n-------------------------------------- Plot error vs lambda --------------------------------------')


print("""\nnow, let's plot error (in mpg) versus regularization parameter
    for various feature sets and polynomial transformation orders.""")

print("""\nas you can see the plot, we want to mention a few point:

    * minimum error or the best prediction is related to:
        - feature set 0 which cylinders feature encoded as one-hot
        - polynomial transformation order 2
        - without regularization parameter.
    * encoding cylinders feature as one-hot works better than standardizing it.
    * poly. order 2 works better than order 1 in every reg. params.
    * for poly. order 1 and 2, the best reg. param. is 0, so in these cases
        structural error (bias) is dominant over estimation error (variance).
        I want to point out that it's a good way to find out the bias variance trade-off with
        increasing reg. param. and check the increase or decrease of evaluation error.
        since the increase of reg. param., increases the structural error (bias) and decreases the estimation error (variance).
    * for poly. order 3, small reg. params. have a high error and by increasing reg. param.
        we (suddenly) obtain lower errors which means estimation error (variance) was high versus structural error (bias)
        and there is a risk of overfitting in low reg. params.
    * even poly. order 1 works better than order 3, which means the auto data is simple enough to model
        and there is no need to model it with order 3.
    """)

Ks = [[1, 2], [3]]
fig, axs = plt.subplots(2, 2)
for featfuncs_num in [0, 1]:
    for Kpos, Klst in enumerate(Ks):
        for K in Klst:
            d = err_dict[featfuncs_num][K]
            x, y = zip(*sorted(d.items()))
            ax = axs[featfuncs_num, Kpos]
            ax.plot(x, y, label=f'K = {K}')
            ax.legend()
            ax.set_title(f'feature set {featfuncs_num}')
        
        # we can annotate after completely constructing each axes
        # because xlim and ylim is now determined and fixed
        for K in Klst:
            min_err, min_comb = min_finder(err_dict, [featfuncs_num], [K])
            xy = (min_comb[-1], min_err)
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            coef = -1 if K == 1 else 1
            xytext = ((xy[0]-xlim[0])/(xlim[1]-xlim[0]) + 0.1, (xy[1]-ylim[0])/(ylim[1]-ylim[0]) + coef * 0.1)
            # annotation
            ax.annotate(f'minimum in {(min_comb[-1], round(min_err, 3))}', xy=xy, xytext=xytext, textcoords='axes fraction',
                arrowprops=dict(width=0.5, headwidth=5, shrink=0.05))

fig.supxlabel('lambda')
fig.supylabel('error (in mpg)')
fig.tight_layout()
plt.pause(1)
input('\ncontinue and close the plot?')




print('\n--------------------------------- What if we consider model_year feature? ---------------------------------')


print("""\nwe add model_year feature (standardized) to the features.
    based on the results obtained in previous sections. we guess the best
    poly. trans. order should be 2 and the best reg. params. in this transformation should be 0.
    let's evaluate the new feature set in this case.""")

featfuncs2 = [('cylinders', hw5.one_hot),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot),
            ('model_year', hw5.standard)]   # model_year added to features
K = 2
lam = 0

auto_datapts_mdlyr, _ = hw5.auto_data_and_targets(auto_data_dict, featfuncs2)
poly_transformer = hw5.make_poly_trans_func(K)
auto_datapts_mdlyr_trans = poly_transformer(auto_datapts_mdlyr)
auto_datapts_mdlyr_trans = del_poly_augment(auto_datapts_mdlyr_trans)

std_error_mdlyr = hw5.xval_ridge_reg(auto_datapts_mdlyr_trans, auto_targets, lam)
print('\nerror in this case:', round(std_error_mdlyr * trg_sigma, 3))
print('which gives us a better prediction compared with feature set 1 which had error:', round(min_error_all, 3))

input('\ncontinue?')



print('\n------------------------------------ feature set which car_name added ------------------------------------')


print("""\nnow we want to consider textual data in car_name field in the features of the datapoints.""")


featfuncs3 = [('cylinders', hw5.one_hot),
            ('displacement', hw5.standard),
            ('horsepower', hw5.standard),
            ('weight', hw5.standard),
            ('acceleration', hw5.standard),
            ('origin', hw5.one_hot),
            ('model_year', hw5.standard)]
print('\nfeature set 3 (except car_name, which will be considered soon) are consist of the followings:\n',
    [(feature_name, feature_func.__name__) for feature_name, feature_func in featfuncs3])
print('\nfirst of all we construct features with feature set introduced above.')
auto_datapts_without_text, _ = hw5.auto_data_and_targets(auto_data_dict, featfuncs3)
print('\nauto datapoints without car_name is constructed and has the shape (d1, n):', auto_datapts_without_text.shape)
print('\nyou can see the first datapoint feature vector which arranged in a row below:\n',
    auto_datapts_without_text[:,:1].T)


print("""\nin the next step we construct bow representation of car_name textual data
    and extend datapoints coordinates (featuer vectors) by this feature representation.""")
data_dict = hw5.load_text_data(path_auto_data)
data_texts = hw5.text_data_dict_to_list(data_dict)
dictionary = hw5.bag_of_words(data_texts)
data_texts_array = hw5.extract_bow_feature_vector(data_texts, dictionary)
print("\nnow bow representation of car_name's are ready. Has the shape (d2, n):", data_texts_array.shape)
print("\nand let's see the first datapoint's bow rep. which arrange in a row below:\n",
    data_texts_array[:,:1].T, '\nwhich is corresponding to car_name:', data_texts[0])
print("""\nfor you to be sure of correctness of the respresentation,
    we show you the first 3 dictionary (token, tokenid):\n""",
    sorted(dictionary.items(), key=lambda x: x[1])[:3])


print("\nnow it's time to attach two feature vectors (featfuncs3 and bow rep.) into a whole.")
auto_datapts_with_text = np.vstack((auto_datapts_without_text, data_texts_array))
print("\nafter that, the complete data array has the shape (d1+d2, n):", auto_datapts_with_text.shape)


nbin = 50
lam = 0
max_iter = 50000
print(f"""\nbefore goint for evaluation, we want to be sure about maximum iteration number,
    which is a hyperparameter of the stochastic gradient descent (learning algorithm we use here).
    for this, we draw the convergence plot of two dataset and compare them:
        - without text which is datapoints featured by feature set 3 above
        - with text which is the same without-text features extended by bow representation
    note that the convergence plot is binned in {nbin} bins (each bin is averaged in x and y) and lam is set to {lam}.""")
fig, axes = plt.subplots(2, 1)
datapts_tuplst = [
    ('auto datapoints without text', auto_datapts_without_text),
    ('auto datapoints with text', auto_datapts_with_text),
]
print('\nplots are drawing...')
for (datapts_name, datapts_array), ax in zip(datapts_tuplst, axes):
    hw5.ridge_min(datapts_array, auto_targets, lam, max_iter,
        cvg_plot=True, ax=ax, nbin=50, pause=False)
    ax.set_title(datapts_name)
fig.tight_layout()
print(f"""\nplots are drawn.
    as you can see, two plots have the very same shape such that you can barely notice the differences.
    and you can say that in 10000 iterations we have achieved a good parameter.
    
    note that because of the standardization of features, the minimizer (sgd) will mostly search
    the d-dimensional cube in the range of -1 to +1 in every direction in parameter space.
    
    it's a surprise that sgd for datapoints with and without text feature has the same
    number of iterations to reach an acceptable minimum, because in the with-text case,
    sgd theoretically has a bigger exploration space because of bigger volume of d-dimensional cube.
    
    I think it's because of no any useful information is stored in textual data here,
    so the effective exploration space for sgd is reduced to the first {auto_datapts_without_text.shape[0]} dimensions,
    which are the dimensions of auto datapoints without text feature vectors.
    you can assure yourself by looking at the final parameters of ridge_min and compare the first {auto_datapts_without_text.shape[0]}
    with the rest, which are bigger almost one order in 10.""")
plt.pause(1)
input('\nClose the plots?')


max_iter = 10000
print(f"""\nnow we set sgd's number of iterations to {max_iter} and evaluate ridge regression on with and without text cases,
    with regularization parameter lam = {lam}.""")
print('\nevaluating...')
for datapts_name, datapts_array in datapts_tuplst:
    std_err = hw5.xval_ridge_reg(datapts_array, auto_targets, lam, max_iter)
    print(f'RMSE evaluation error for {datapts_name} (in mpg):', round(std_err * trg_sigma, 3))


print(f"""\nis lambda = {lam} the best reg. param.?
    as we've seen in the previous sections, polynomial transformation order 1 (feature vectors itself)
    does not have enough complexity to regularize. so we can rest assured that lam = {lam} is the best reg. param. here.""")


K = 2
print(f"""\nwhat about other polynomial transformation orders?
    even though we know there is no such an information in textual features,
    I want to bother myself (and the processor more!) to have an evaluation on poly. transformed feature vectors
    in order {K} (and also lam = {lam}).\n""")
poly_transformer = hw5.make_poly_trans_func(K)
print('\nthis may takes one or two minutes...')
for datapts_name, datapts_array in datapts_tuplst:
    datapts_array_transed = poly_transformer(datapts_array)
    print(f'shape of polynomial transformed {datapts_name}:', datapts_array_transed.shape)
    std_err = hw5.xval_ridge_reg(datapts_array_transed, auto_targets, lam, max_iter)
    print(f'RMSE evaluation error for polynomial transformed {datapts_name}:', round(std_err * trg_sigma, 3))
print('\nas you can see, there is no significant difference in errors as I said before.')


input('\nhomework 5 is done.\npress enter to close it.')