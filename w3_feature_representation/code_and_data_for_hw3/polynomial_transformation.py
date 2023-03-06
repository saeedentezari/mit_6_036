#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Polynomial Transformation Implementation by Solving Combinations with Replacement Problem
=========================================================================================

Polynomial transformation should be obtained by finding combinations_with_replacement of size k from a set of items.
This implementation has a dummy approach to solve the problem by representing combinations in terms of powersTuples.

An standard way of finding combinations_with_replacement is
create a generator from itertools.combinations_with_replacement(iterable (of items), k)
which returns combinations as tuples of items themselves (obviously result's sort does not matter in either cases!)

The difference of mine is in the representation of the combinations (and also the runtime on big data),
here I represented each combination as a tuple of 'how many times the items is in the combination'
or as you may say, the 'powers of items' in the polynomial transformation.



Introduction
============

Consider the polynomial transformation as the function below,
which have all the powers of coordinates:

    poly_trans(K, [x1, ..., xd]) = (for K = 2, and d = 2)
        = [x1^0 * x2^0] + [x1^1 * x2^0, x1^0 * x2^1] + [x1^2 * x2^0, x1^1 * x2^1, x1^0 * x2^2]
        = [1] + [x1, x2] + [x1^2, x1x2, x2^2]

    '+' means concatenate here,
    each bracket contains transofrmed features of order k, k from 0 to K.
    K: max order
    d: dimension of raw feature vector
    


What is powersTuple?
====================

For each transformed feature in polynomial basis, we collect the powers of x1 to xd respectively in a tuple.
For example, for d = 2:
    the powerTuple for transformed feature x1x2 is (1, 1),
    which means x1 is powered to 1 and x2 is powered to 1.
    or powerTuple for x1^2 is (2, 0)
The powerTuple is of length 'd' (number of coordinates) and the sum of elements 'k' (order).

"""

import numpy as np



__author__ = "Saeed Entezari"
__email__ = "s.entezari144@gmail.com"



def powersTuples_to_order(K, d):
    '''
    Collect all the powersTuples of orders k in a dictionary, k from 0 to K.
        For avoiding multiple computing of powersTuples,
        we call powersTuples_of_order(k, d) by k = K once,
        and save all the results in respective recursive calls of its run in a dictionary.

        It is an equivalent (but slower) solution of combinations with replacement problem.

    Params:
        K (int): max order K, 0 <= k <= K
        d (int): dimension of raw feature vector

    Returns:
        a dictionary of orders k (int) -> list of powersTuples of order k
    '''

    def powersTuples_of_order(k, d):
        '''
        Give all powerTuples of order k for a d-dimensional raw feature vector,
            in a recursive manner.

        Params:
            k (int): order k
            d (int): dimension of raw feature vector

        Returns:
            a list of all possible powersTuples of order k

        !   SPEED-UP:
                dooplicate check has a significant complexity in big input sizes,
                so it's better to design algorithm differently.
                also notice the copying cost here, which can be omited by representing
                integers instead of tuples, (1, 0, 2) -> 102
                and adding 1, 10, 100, and so on, to integer numbers in operations,
                to change the corresponding value in tuples.
                be careful about iteration over this integer stuff,
                because these are combinations and combinations should be iterable.
        '''
        # base case for k = 0
        if k == 0:
            result = [(0,) * d]
            powTupsOrd[k] = result
            return result

        # otherwise do the recursive solution
        else:
            # loop on order k-1 solutions list
            result = []
            for solution in powersTuples_of_order(k-1, d):
                # add possible elements to the result based on this solution, after dooplicate check

                # loop on coordinates of the solution
                for cor in range(d):
                    # add one to the value of this coordinate
                    added = solution[cor] + 1
                    # make a new possible solution based on the added one and the rest not changed
                    possol = solution[:cor] + (added,) + solution[cor+1:]   ### notice the copying cost
                    # doolicate check                                       ### notice the dooplicate check cost on big data!
                    if possol not in result:
                        # save the verified possible to the result
                        result.append(possol)

            # save the result of this call (k) into a dictionary
            powTupsOrd[k] = result
            # return the result
            return result

    # call powersTuples_of_order once and save the result
    # into a dictionry of orders (int) -> powersTuples (list of powersTuple)
    powTupsOrd = {}
    powersTuples_of_order(K, d)
    return powTupsOrd

# # test case
# K, d = 2, 2
# print(powersTuples_to_order(K, d))


def polyfeat(datapoint, powersTuple):
    '''
    Calculate the (value of) polynomial feature based on datapoint and powersTuple given.

    >>> polyfeat([x1, x2], (3, 4))
    x1^3 * x2^4

    Params:
        datapoint (dx1 array)
        powersTuple (tuple with length d and sum k)
    
    Returns:
        polynomial feature calculated (int)
    '''
    return np.prod(datapoint**powersTuple)


def make_poly_trans_func(K):
    '''
    Wrap polynomial transformation function for max order K given.
    '''
    def poly_trans(raw_features):
        '''
        Transforms raw feature vectors consisting of n d-dimensional raw feature vector
            in polynomial basis to order K.

        Params:
            raw_feature (dxn array)

        Returns:
            transformed feature vectors (Sxn array)
                S is the sum of the number of polynomial features to max order K.
        '''
        d, n = raw_features.shape
        # make the dictionary of orders -> powersTuples for later use
        powTupsOrd = powersTuples_to_order(K, d)
        result = []
        # loop over each datapoint (column) in raw_features
        for j in range(n):
            transfeat = []
            # loop over orders from 0 to K
            for k in range(K+1):
                # loop over all powersTuple
                for powersTuple in powTupsOrd[k]:
                    # compute the polynomial feature with respect to datapoint and this powersTuple
                    # then append it to polynomial features vector
                    transfeat.append(polyfeat(raw_features[:,j], powersTuple))
            # append the transformed features vector to the result
            # result is a list of transformed columns of raw_feature (input)
            result.append(np.array(transfeat).reshape(-1, 1))
        # make an array out of the result (list of tranformed columns)
        return np.hstack(result)

    return poly_trans


if __name__ == '__main__':

    K = 2
    raw_features = np.array([[2, 4], [3, 5]])
    poly_trans_func = make_poly_trans_func(K)
    print('Raw features as input:\n', raw_features)
    print(f'Transformed features in polynomial basis to order K = {K}:\n',
            poly_trans_func(raw_features))