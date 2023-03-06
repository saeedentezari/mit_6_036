#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Batch Normalization Module
--------------------------

Implementation of batch normalization module.

Test is included.
"""

import numpy as np


__author__ = "Saeed Entezari"
__email__ = "s.entezari144@gmail.com"



class Module:
    '''
    A parental class for linear and non-linear modules to ensure
        each module has a method sgd_step to call.
        If sgd_step is not defined (in non-linear modules), its call should be passed.
    '''
    def sgd_step(self, lrate): pass     # for modules w/o weights



class BatchNorm(Module):
    '''
    Batch normalization module
    --------------------------

    Initiated by number of inputs m (= # outputs, n) to construct initial parameters of
        scale (G, m x 1) and shift (B, m x 1).
        and an eps value to avoid dividing by zero in calculations when the var is zero.


    forward method:
        takes in a mini-batch of Z values by size m x K, K m-dimensional datapoints arranged in the columns,
        extract the average and variance of the columns as mu and var m x 1 vectors,
        standardize Z by mu and var, to get Zbar (m x K) which each coordination (row) of it has avg 0 and var 1,
        then scale and shift the values of Zbar by G and B to get the m x K output array Zhat = G * Zbar + B.

        so in a nutshell:   Z    ->    Zbar = (Z - mu) / sqrt(var)    ->    Zhat = G * Zbar + B

        we store some obtained values of the forward to use in backward, such as: Z, K, mu, var.
        and Zbar for sgd_step.


    backward method:
        takes in a mini-batch of dLdZhat by size m x K, and to obtain the dLdZ we notice that:

        dLdZbar_j = G * dLdZhat_j,
        dLdvar = sum_k over dLdZbar_k * (Z_k - mu) * -0.5 (var+eps)**-3/2,
        dLdmu = sum_k over dLdZbar_k * -1/(var+eps)**0.5 + dLdvar * sum_k over -2(Z_k - mu)/K,

        dLdZ_j = dLdZbar_j * 1/(var+eps)**0.5 + dLdvar * 2(Z_k - mu)/K + dLdmu / K,

        where the subscript j indicates the datapoint in column j of the array.
        you can see more details in the Ioffe and Szegedy, 2015 paper in the link below:
        https://arxiv.org/pdf/1502.03167.pdf


    sgd_step method:
        notice that BatchNorm module has two parameters G and B, scale and shift,
        and the gradient of loss with respect to each one of them can be obtained by:

        dLdG = sum_k over dLdZhat_k * Zbar_k,
        dLdB = sum_k over dLdZhat_k.

        we calculate the above gradients in the backward method to avoid from storing unnecessary values.
    '''
    def __init__(self, m):

        self.m = m                                          # number of inputs (or output units)
        self.B = np.zeros([m, 1])                           # shift params
        self.G = np.random.normal(0, m ** -0.5, [m, 1])     # scale params
        self.eps = 1e-20                                    # epsilon


    def forward(self, Z):

        self.Z = Z
        self.K = Z.shape[1]

        self.mu = np.mean(Z, axis=1, keepdims=True)
        self.var = np.mean((Z - self.mu)**2, axis=1, keepdims=True)

        self.Zbar = (self.Z - self.mu) / np.sqrt(self.var)
        return self.G * self.Zbar + self.B                  # return Zhat


    def backward(self, dLdZhat):

        # re-usable constants
        std_inv = 1 / np.sqrt(self.var + self.eps)          # inverse of the standard deviation
        Z_min_mu = self.Z - self.mu                         # Z - mu

        # dLdZbar
        dLdZbar = self.G * dLdZhat
        # dLdvar
        dLdvar = np.sum(dLdZbar * Z_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True)
        # dLdmu
        dLdmu = np.sum(dLdZbar * -std_inv, axis=1, keepdims=True) + dLdvar * -2/self.K * np.sum(Z_min_mu, axis=1, keepdims=True)

        # dLdZ
        dLdZ = (dLdZbar * std_inv) + (dLdvar * 2 * Z_min_mu / self.K) + (dLdmu / self.K)

        # now that we have the partial derivatives, we can calculate dLdG and dLdB for sgd_step
        self.dLdG = np.sum(dLdZhat * self.Zbar, axis=1, keepdims=True)
        self.dLdB = np.sum(dLdZhat, axis=1, keepdims=True)

        return dLdZ


    def sgd_step(self, lrate):
        
        self.G = self.G - lrate * self.dLdG
        self.B = self.B - lrate * self.dLdB


    # setters to use in batchnorm_test()
    def set_scale(self, G):
        self.G = G
    
    def set_shift(self, B):
        self.B = B
    
    def set_epsilon(self, eps):
        self.eps = eps


#######################################################################################
# Test BatchNorm Module
#######################################################################################

def expect_test(unit_name, expected, actual):
    '''
    A function to compare expected and actual result.
    '''
    if np.allclose(expected, actual):
        print(unit_name + ' as expected: PASSED.')
    else:
        print(unit_name + ' unexpected: FAILED!')
        print('expected:\n', expected)
        print('but the actual:\n', actual)


def batchnorm_test():

    print('\nTesting BatchNorm module...\n')
    
    # create batch normalizer module and set scale, shift and eps parameters
    bn = BatchNorm(2)
    G, B, eps = np.array([[1, 2]]).T, np.array([[-1, 1]]).T, 0
    bn.set_scale(G); bn.set_shift(B); bn.set_epsilon(eps)

    # forward test
    Z = np.array([[-1, 1], [0, 4]])                     # input Z of forward
    Zhat_exp = np.array([[-2, 0], [-1, 3]])             # output calculated by hand
    expect_test('in forward:', Zhat_exp, bn.forward(Z))

    # backward test
    dLdZhat = np.array([[4, 10], [8, 16]])              # input dLdZhat of backward
    dLdZ_exp = np.array([[0, 0], [0, 0]])               # output calculated by hand
    expect_test('in backward:', dLdZ_exp, bn.backward(dLdZhat))

    # sgd_test
    lrate = 1
    bn.sgd_step(lrate)
    G_exp, B_exp = np.array([[-5, -6]]).T, np.array([[-15, -23]]).T
    expect_test('G after sgd_step:', G_exp, bn.G)
    expect_test('B after sgd_step:', B_exp, bn.B)

# batchnorm_test()