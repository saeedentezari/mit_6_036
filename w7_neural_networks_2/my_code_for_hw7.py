#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Linear, Non-Linears, Loss, and Sequential Modules as NN
-------------------------------------------------------

Sequential module combines a sequence of linear and non-linear modules which ends with the loss module
    to construct a Neural Network structure.
All the modules have forward, backward and sgd methods for training process.

Modules can work with mini-batches and perform a simple sgd with constant learning rates during training.

Tanh(), ReLU(), and SoftMax() non-linear modules, and a NLL() loss module are prepared.

Tests are included.
"""



import numpy as np
import matplotlib.pyplot as plt
from my_expected_results import *           # test values
from my_code_for_hw3_part1copy import *     # for plotting, copy my_code_for_hw3_part1.py to the file directory



__author__ = "Saeed Entezari"
__email__ = "s.entezari144@gmail.com"




class Module:
    '''
    A parental class for linear and non-linear modules to ensure
        each module has a method sgd_step to call.
        If sgd_step is not defined (in non-linear modules), its call should be passed.
    '''
    def sgd_step(self, lrate): pass     # for modules w/o weights


class Linear(Module):
    '''
    Linear Modules

    Initiate with number of inputs m, and number of outputs n.

    Each linear module has a forward method that takes in a batch of activations A
        (from the previous layer) and returns a batch of pre-activations Z.
        It also stores A as an atribute of this object for use in backward call.

    Each linear module also has a backward method that takes in (a batch of) dLdZ
        and returns (a batch of) dLdA (for the previous layer).
        Here we can compute and store dLdW and dLdW0 of the batch for later use in sgd_step call.

    This class also has a sgd update method sgd_step which updates weights attribute
        of the module object.
    '''

    def __init__(self, m, n):
        self.m, self.n = m, n   # (input size, output size) or (size of prev A vectors, size of Z vectors)
        self.W0 = np.zeros([n, 1])  # a (n x 1) vector of offset (bias)
        self.W = np.random.normal(0, m ** (-0.5), [m, n])   # (m x n) matrix of weights with mean 0 and std 1/sqrt(m)

    def forward(self, A):
        self.A = A      # (m x b) a batch of A from previous layer
        return self.W.T @ A + self.W0  # (n x b) a batch of Z in this layer

    def backward(self, dLdZ):   # dLdZ is (n x b)
        # now we have dLdZ, we can compute and store dLdW and dLdW0 on the batch
        self.dLdW = self.A @ dLdZ.T     # uses stored attribute self.A, notice the beauty of matrix product
                                        # in the view of sum of the corresponding columnA x rowB products
        self.dLdW0 = np.sum(dLdZ, axis=-1, keepdims=True)
        # then return dLdA for prev layer
        return self.W @ dLdZ    # dLdA (m x b)

    def sgd_step(self, lrate):
        self.W = self.W - lrate * self.dLdW
        self.W0 = self.W0 - lrate * self.dLdW0

    def set_weights_W(self, W):
        self.W = W



class Tanh(Module):
    '''
    Tanh activation module

    Each Tanh activation module has a forward method that takes in a batch of pre-activations Z
        and returns a batch of activations A = Tanh(Z). It also stores A to use later in backward.

    Also has a backward method that takes in a batch of dLdA and returns dLdZ.

    When sgd method is called on a sequence of linear and non-linear modules as a NN object,
        then Tanh module will be called by sgd_step method that should be passed,
        because there is no weight in this object.
        Since we have various non-linear activation modules, we define a parental class named Module
        that has sgd_step method which only passes the call.
    '''
    def forward(self, Z):   # (n x b) a batch of pre-activations
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):   # (n x b) a batch of dLdA
        # note that dLdZ = tanh'(Z) * dLdA, which tanh'(Z) = 1 - tanh(Z)**2 = 1 - A**2
        return (1 - self.A**2) * dLdA



class ReLU(Module):
    '''
    ReLU activation module

    Each ReLU activation module has a forward method that takes in a batch of pre-activations Z
        and returns a batch of activations A = ReLU(Z). It also stores A to use later in backward.

    So also has a backward method that takes in a batch of dLdA and returns dLdZ.
    '''
    def forward(self, Z):   # (n x b) a batch of pre-activations
        self.A = np.maximum(Z, 0)
        return self.A

    def backward(self, dLdA):   # (n x b) a batch of dLdA
        return dLdA * (self.A != 0)



class SoftMax(Module):
    '''
    SoftMax activation module

    Each SoftMax activation module has a forward method that takes in a nxb batch of pre-activation Z
        and returns a nxb batch of activations A = SoftMax(Z), which components Aij = exp(Zij) / sum(Zj),
        and can be interpreted as probabilities (for classification into n classes).
        Hence this module will be used as the last layer of a NN solving cassification problem.

    And also has a backward method that takes in dLdZ directly from Negative Log-Likelihood (or Cross-Entropy)
        because it's easy to calculate
        (dLdZ in the last layer as softmax by NLL loss = A - Y,
        just like the linear activation function along with square differences loss)
        and NLL has the value Y and input A stored.
    '''
    def forward(self, Z):   # (n x b) batch of pre-activations
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def backward(self, dLdZ):   # dLdZ will be given to SoftMax directly by NLL
        return dLdZ             # just pass the value

    def class_fun(self, Ypred):
        # takes a batch of activations in the last layer (Ypred)
        # and returns a (1 x b) vector of classes indecies based on prediction Ypred
        return np.argmax(Ypred, axis=0, keepdims=True)



class NLL(Module):
    '''
    Negative log-likelihood loss module

    Has a forward method that takes in:
        output Ypred (n x b), a batch of activations from last layer (that should be SoftMax to match)
        and batch of corresponding one-hot encoded labels Y as a (n x b) array,
        then returns loss on these predictions as a scalar which is sum of losses on each pair Ypred and label Y.
        Forward method also stores Ypred and label Y as the attributes to use in backward.

    Has a backward method that uses stored Ypred and label Y, and returns dLdZ in the last layer (SoftMax module) directly.
        Notice that SoftMax's backward is constructed such that takes dLdZ directly and immediately returns it
        (to previous layer in backward method of Sequential NN).
    '''
    def forward(self, Ypred, Y):
        self.Ypred = Ypred      # (n x b) a batch of softmax output prediction
        self.Y = Y              # (n x b) one-hot labels in columns
        return -np.sum(Y * np.log(Ypred))   # batch loss as a scalar

    def backward(self):
        return self.Ypred - self.Y  # returns dLdZ of SoftMax (with NLL loss) directly = A - Y (nxb array)
        # notice that dLdZ of identity activation function with squared loss is also A - Y



class Sequential:
    '''
    Sequential linear/non-linear and loss modules as a Neural Network

    Initialize by a list of linear/non-linear modules and a loss module that matches the last non-linear module.

    forward method:
        that takes in a batch of datapoints X
        and pass it through all the modules by calling their forward method respectively,
        and returns the output of the last module of the last layer which is the output of the NN.

    backward method:
        that takes in a batch of dLdZ from the loss module
        (note that in this implementation, the last layer always has a SoftMax module matched by NLL loss - classification,
        and we give the dLdZ, not dLdA, to SoftMax by the NLL loss directly,
        and SoftMax just passes it to the linear module in the last layer).
        This method passes (a batch of) gradients of loss with respect to the pre-activations or activations (dLdZ or dLdA)
        through all the network in the backward direction by calling backward method of all the modules.
        It also compute dLdW and dLdW0 of each linear module based on the batch given in forward,
        and note that it returns nothing.

    sgd_step method:
        will be called on all modules (except loss module) to update their weights and offsets (by a specified lrate)
        based on dLdW and dLdW0 computed for each linear module on the backward method.
        Non-linear modules just pass (ignore) the sgd_step call, because there is no weight to update.

    sgd method:
        takes in the whole dataset datapoints X and their labels Y, learning rate, number of iterations, and batch size.
        randomly pick a batch of datapoints and their labels from dataset,
        does a sequential forward method followed by a forward on loss module,
        then a backward on loss followed by a sequential backward method,
        and at the end, call sgd_step to update all the parameters of NN by the specified learning rate.
        Repeat the whole process by the number of iterations given.
    '''
    def __init__(self, modules, loss):  # list of modules, loss module
        self.modules = modules
        self.loss = loss

    def forward(self, Xt):   # forward NN and compute Ypred
        for m in self.modules: Xt = m.forward(Xt)
        return Xt   # returns Ypred

    def backward(self, delta):  # backward NN and compute dLdW and dLdW0 for linear modules
        # delta refers to dLdA and dLdZ both
        # note the reversed order of modules
        for m in self.modules[::-1]: delta = m.backward(delta)

    def sgd_step(self, lrate):  # gradient descent step, update NN parameters
        for m in self.modules: m.sgd_step(lrate)


    def sgd(self, X, Y, iters=100, lrate=0.005, batch_size=1,
                print_accuracy=False, plot_loss=False, plot_classifier=False):      # train
        D, N = X.shape
        assert batch_size <= N  # batch_size can be at most to the number of datapoints
        self.loss_values = []   # define an attribute to track loss values in training
        # prepare data axis to plot in every iterations
        if plot_classifier: data_axis = plot_data(X, Y); data_axis.set_title('training data')
        for it in range(iters):
            # pick randomly a batch of datapoints and their labels 
            idx = np.random.choice(range(N), size=batch_size, replace=False)
            Xt, Yt = X[:, idx], Y[:, idx]
            # forward
            Ypred = self.forward(Xt)
            batch_loss = self.loss.forward(Ypred, Yt)
            # backward
            delta = self.loss.backward()
            self.backward(delta)
            # sgd_step params update
            self.sgd_step(lrate)
            # calculating total loss by forwarding the whole dataset through the NN
            if plot_loss: self.loss_values.append(self.current_loss(X, Y))
            # print accuracy once a 20 steps
            if print_accuracy: self.print_accuracy(it, X, Y, self.current_loss(X, Y), every=iters//20)
            # plot classifier once a 20 steps
            if plot_classifier and it % (iters//20) == 0:
                def predictor(x, y):
                    X = np.array([[x], [y]])
                    return 2 * self.forward(X)[0, 0] - 1
                plot_nonlinear_separator(predictor, data_axis)
                plt.pause(0.01)
                # input('Next step?')
        # plot loss
        if plot_loss: self.plot_loss()

    def print_accuracy(self, it, X, Y, cur_loss, every=250):
        # print the accuracy (# correct predictions / # all precitions) and the loss
        if it % every == 0:
            Ypred = self.forward(X)
            cf = self.modules[-1].class_fun     # assume the last module is SoftMax
            acc = np.mean(cf(Ypred) == for_softmax_invert(Y))
            print('on Iteration =', it, '\tAccuracy =', acc, 'Loss =', cur_loss, flush=False)

    def current_loss(self, X, Y):
        # calculate current loss on the whole dataset
        Ypred = self.forward(X)
        return self.loss.forward(Ypred, Y)

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.loss_values)
        ax.set_xlabel('number of iteration (or epoch)')
        ax.set_ylabel('loss')
        plt.pause(0.01)
        input('Close the plot?')




#####################################################################################
# Data Sets
#####################################################################################


def for_softmax(y):
    '''
    Convert 1 x N array of labeles consists of class numbers 0 and 1
        into 2 x N array of one-hot encoded labels.
    '''
    return np.vstack([1 - y, y])


def for_softmax_invert(y):
    '''
    Do the inverse of for_softmax:
        Convert 2 x N array of one-hot encoded labels
        into 1 x N array of labels consists of class numbers 0 and 1.
    '''
    return 1 - y[0:1, :]


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)


def hard():
    X = np.array([[-0.23390341, 1.18151883, -2.46493986, 1.55322202, 1.27621763,
                   2.39710997, -1.3440304, -0.46903436, -0.64673502, -1.44029872,
                   -1.37537243, 1.05994811, -0.93311512, 1.02735575, -0.84138778,
                   -2.22585412, -0.42591102, 1.03561105, 0.91125595, -2.26550369],
                  [-0.92254932, -1.1030963, -2.41956036, -1.15509002, -1.04805327,
                   0.08717325, 0.8184725, -0.75171045, 0.60664705, 0.80410947,
                   -0.11600488, 1.03747218, -0.67210575, 0.99944446, -0.65559838,
                   -0.40744784, -0.58367642, 1.0597278, -0.95991874, -1.41720255]])
    y = np.array([[1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1.,
                   1., 0., 0., 0., 1., 1., 0.]])
    return X, for_softmax(y)


def xor():
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, 0, 0]])
    return X, for_softmax(y)


#####################################################################################
# Tests
#####################################################################################


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



def lin_module_test(test_values):
    '''
    Test linear module implementation by test values as a dictionary.
    '''
    print('\nTesting Linear module in forward, backward and sgd step...')
    m, n = test_values['m'], test_values['n']
    lin = Linear(m, n)
    lin.set_weights_W(test_values['W_set'])
    
    # forward test
    Z = lin.forward(test_values['input']['A'])
    expect_test('linear_forward: Z', test_values['expect']['Z'], Z)

    # backward test
    dLdA = lin.backward(test_values['input']['dLdZ'])
    expect_test('linear_backward: dLdA', test_values['expect']['dLdA'], dLdA)
    dLdW = lin.dLdW
    dLdW0 = lin.dLdW0
    expect_test('linear_gradient: dLdW', test_values['expect']['dLdW'], dLdW)
    expect_test('linear_gradient: dLdW0', test_values['expect']['dLdW0'], dLdW0)

    # sgd_step test
    lrate = test_values['lrate']
    lin.sgd_step(lrate)
    Wt, W0t = lin.W, lin.W0
    expect_test('linear_param_update: Wt', test_values['expect']['Wt'], Wt)
    expect_test('linear_param_update: W0t', test_values['expect']['W0t'], W0t)

# lin_module_test(lin_module_test_values)



def tanh_module_test(test_values):
    '''
    Test tanh module implementation by test values as a dictionary.
    '''
    print('\nTesting Tanh module in forward and backward and sgd step call...')
    f = Tanh()

    # forward test
    Z = test_values['input']['Z']
    expect_test('tanh_forward: A', test_values['expect']['A'], f.forward(Z))

    # backward test
    # the same A that f produced in forward test is considered
    dLdA = test_values['input']['dLdA']
    expect_test('tanh_backward: dLdZ', test_values['expect']['dLdZ'], f.backward(dLdA))

    # sgd_step test
    # should return None, because calling f by sgd_step method should pass
    lrate = 1
    if f.sgd_step(lrate) is None:
        print('tanh_sgd_step: passes as expected: PASSED.')
    else:
        print('tanh_sgd_step: doesn\'t pass: FAILED!')
        print('sgd_step return value\n', f.sgd_step(lrate))

# tanh_module_test(tanh_module_test_values)



def relu_module_test(test_values):
    '''
    Test relu module implementation by test values as a dictionary.
    '''
    def mutation_check(unit_name, before, after):
        if np.allclose(before, after):
            print(unit_name + ' not mutated: OK.')
        else:
            print(unit_name + 'mutated: NOT OK!')
            print('before:\n', before)
            print('after:\n', after)

    print('\nTesting ReLU module in forward and backward...')
    f = ReLU()

    # forward test + mutation check of input Z
    Z = test_values['input']['Z']
    Zbef = Z.copy()
    expect_test('relu_forward: A', test_values['expect']['A'], f.forward(Z))
    Zaft = Z
    mutation_check('input Z to forward method', Zbef, Zaft)

    # backward test + mutation check of attribute A
    dLdA = test_values['input']['dLdA']
    Abef = f.A.copy()
    expect_test('relu_backward: dLdZ', test_values['expect']['dLdZ'], f.backward(dLdA))
    Aaft = f.A
    mutation_check('attribute A in backward call', Abef, Aaft)

# relu_module_test(relu_module_test_values)



def softmax_module_test(test_values):
    '''
    Test softmax module implementation by test values as a dictionary.
    '''
    print('\nTesting SoftMax module in forward and backward...')
    f = SoftMax()

    # forward test + check for all activations sums to 1
    Z = test_values['input']['Z']
    Ypred_forw = f.forward(Z)
    expect_test('softmax_forward: Ypred', test_values['expect']['Ypred'], Ypred_forw)
    if np.all(np.sum(Ypred_forw, axis=0) == 1):
        print('all activations sums to 1: OK.')
    else:
        print('there is an activation which doesn\'t sum to 1: FAILED!')
        print('sum for each activation produced in forward\n', np.sum(Ypred_forw, axis=0))

    # class_fun test
    # use the transpose of Ypred prepared in test_values
    Ypred = test_values['expect']['Ypred'].T
    expect_test('softmax_class_fun: predicted class indecies', test_values['expect']['classes'], f.class_fun(Ypred))

# softmax_module_test(softmax_module_test_values)



def nll_module_test(test_values):
    '''
    Test nll module implementation by test values as a dictionary.
    '''
    print('\nTesting NLL module in forward and backward...')
    loss = NLL()

    # forward test + check loss value type
    Ypred = test_values['input']['Ypred']
    Y = test_values['input']['Y']
    loss_value_forw = loss.forward(Ypred, Y)
    if isinstance(loss_value_forw, float):
        print('calculated loss is float: OK.')
    else:
        print('calculated loss is not float: FAILED!')
        print('it has the type', type(loss_value_forw))
    expect_test('nll_forward: loss', test_values['expect']['loss'], loss_value_forw)

    # backward test
    expect_test('nll_backward: dLdZ', test_values['expect']['dLdZ'], loss.backward())

# nll_module_test(nll_module_test_values)



def sequential_module_test(test_values):
    '''
    Test Sequential module implementation in forward, backward and sgd
        by a simple NN created to test and a super simple dataset,
        and compare the result with the test_values dictionary.
        Note that Linear, ReLU, SoftMax and NLL module should be defined.
    '''
    print('\nTesting Sequential module in forward, backward and sgd...')
    # dataset
    X, Y = super_simple_separable()
    # before creating sequential module NN, we set a seed
    # to have the same initial weights and pick the batch as we expect
    np.random.seed(0)
    # initializing NN instance
    nn = Sequential([Linear(2, 3), ReLU(), Linear(3, 2), SoftMax()], NLL())
    lin1, f1, lin2, f2 = nn.modules
    loss = nn.loss

    # check initial weights
    # note that W0's initiated by all zeros in Sequential module
    print('\ninitial weights of NN should be matched...')
    expect_test('initial linear1 weights: lin1_W', test_values['lin1_W'], lin1.W)
    expect_test('initial linear2 weights: lin2_W', test_values['lin2_W'], lin2.W)

    # call the sgd method (train) on NN by 1 iteration to get the actual results on modules stored
    nn.sgd(X, Y, iters=1, batch_size=2)


    # now look at the stored attributes of the modules #

    # forward test
    print('\nforward test: checking all activations (the modules just store activations not pre-activations)...')
    # note that A_1 is the input to lin2 and stored in it
    expect_test('A_1', test_values['A_1'], lin2.A)
    expect_test('A_2 (or Ypred)', test_values['A_2'], loss.Ypred)

    # loss in forward and backward
    print('\ntesting loss module in forward and backward...')
    expect_test('loss in forward: loss value', test_values['loss'], loss.forward(loss.Ypred, loss.Y))
    expect_test('loss in backward: dLdZ of last layer', test_values['dLdZ_2'], loss.backward())

    # it's sufficient just to check updated linear module weights to ensure it's working
    print('\nchecking updated parameters...')
    expect_test('updated linear1 weights: upd_lin1_W', test_values['upd_lin1_W'], lin1.W)
    expect_test('updated linear1 offsets: upd_lin1_W0', test_values['upd_lin1_W0'], lin1.W0)
    expect_test('updated linear2 weights: upd_lin2_W', test_values['upd_lin2_W'], lin2.W)
    expect_test('updated linear2 offsets: upd_lin2_W0', test_values['upd_lin2_W0'], lin2.W0)

# sequential_module_test(sequential_module_test_values)






if __name__ == '__main__':
    # create neural network instance
    nn = Sequential([Linear(2, 6), ReLU(), Linear(6, 2), SoftMax()], NLL())
    # nn = Sequential([Linear(2, 10), ReLU(), Linear(10, 10), ReLU(), Linear(10,2), SoftMax()], NLL())

    # choose the training dataset
    X, Y = xor(); iters = 1000
    # X, Y = hard(); iters = 10000
    
    # training
    nn.sgd(X, Y, iters=iters, batch_size=2, print_accuracy=True, plot_loss=True, plot_classifier=True)