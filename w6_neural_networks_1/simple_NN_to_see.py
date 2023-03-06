
import numpy as np
from my_code_for_hw02copy import tidy_plot, plot_separator, plot_data   # copy my_code_for_hw02.py to the file directory
from my_code_for_hw3_part1copy import plot_nonlinear_separator          # copy my_code_for_hw3_part1.py to the file directory
import matplotlib.pyplot as plt


def ReLU(Z):

    res = Z.copy()  # avoid mutation
    res[res < 0] = 0
    return res


def dReLU(Z):

    res = Z.copy()  # avoid mutation
    res[res > 0] = 1
    res[res < 0] = 0
    return res


def sigmoid(Z):

    return 1 / (1 + np.exp(-Z))


def dsigmoid(Z):

    return np.exp(-Z) * sigmoid(Z)**2


def hinge_loss(A, y):

    res = 1 - y * A
    res[res < 0] = 0
    return res


def d_hinge_loss_dA(A, y):

    res = hinge_loss(A, y)
    res[res > 0] = -y
    return res


def squareloss(A, y):

    return np.sum((A - y)**2)


def d_squareloss(A, y):

    return 2 * (A - y)


def NLL(guess, actual):

    return - actual * np.log(guess) - (1 - actual) * np.log(1 - guess)


def dNLL(guess, actual):

    return - actual / guess + (1 - actual) / (1 - guess)



class NN():

    def __init__(self, m1, nls, fls, dfls, loss, dloss):
        
        self.L = len(nls)
        self.nls = nls
        self.mls = (m1,) + nls[:-1]
        self.fls = fls
        self.dfls = dfls
        # self.step_size_fn = step_size_fn

        self.Wls = [np.random.normal(0, 1/ml, (ml, nl)) for ml, nl in zip(self.mls, self.nls)]
        self.W0ls = [np.random.normal(0, 1, (nl, 1)) for nl in self.nls]

        # set weights for test purpose
        # self.Wls = [np.zeros((ml, nl)) for ml, nl in zip(self.mls, self.nls)]
        # for n, Wl in enumerate(self.Wls):
        #     for j, row in enumerate(Wl):
        #         for i, val in enumerate(row):
        #             Wl[i, j] = j - i
        #     self.Wls[n] = Wl
        # self.Wls[-1][1, 1] = 0.5
        # self.W0ls = [np.arange(1, nl+1).reshape(nl, -1) for nl in self.nls]


        self.Zls = [np.zeros((nl, 1)) for nl in self.nls]
        # self.Als = [np.zeros((nl, 1)) for nl in self.nls]
        self.Als = self.Zls.copy()

        self.loss = loss
        self.dloss = dloss


    def get_weights(self, silence = True):

        if not silence:
            print('\nnotice that layers are indexed from 0 to L-1')
            for l in range(self.L):
                print(f'\nlayer {l}:\nW_{l} =', self.Wls[l], f'\n& W0_{l} =', self.W0ls[l])
        return self.Wls, self.W0ls


    def get_Zls(self):
        
        print('\nnotice that layers are indexed from 0 to L-1')
        for l in range(self.L):
            print(f'\nZ_{l} =', self.Zls[l])
        return self.Zls


    def get_Als(self):

        print('\nnotice that layers are indexed from 0 to L-1')
        for l in range(self.L):
            print(f'\nA_{l} =', self.Als[l])
        return self.Als


    def get_loss(self):
        
        return self.loss


    def get_dloss_lastA(self):

        return self.dloss


    def forward(self, x, y):

        # its standard to use getters instead of calling attributes directly
        # but because of print statement we placed in getters, for now we call them directly
        for l in range(self.L):
            self.Zls[l] = self.Wls[l].T @ self.Als[l-1] + self.W0ls[l] if l > 0 else self.Wls[l].T @ x + self.W0ls[l]
            self.Als[l] = self.fls[l](self.Zls[l])

        # and record loss

    
    def backward(self, x, y, lr):

        # getters are skipped for now

        dloss_dlastA = self.get_dloss_lastA()

        newWls = self.L * [0]
        newW0ls = newWls.copy()
        for l in reversed(range(self.L)):

            # error back propagation
            dloss_dAl = self.Wls[l+1] @ dloss_dZl if l < self.L-1 else dloss_dlastA(self.Als[l], y)
            dloss_dZl = np.diagflat(self.dfls[l](self.Zls[l])) @ dloss_dAl  # can be done by just * operation

            # compute gradient with respect to weights
            dloss_dWl = self.Als[l-1] @ dloss_dZl.T if l > 0 else x @ dloss_dZl.T
            dloss_dW0l = dloss_dZl

            # sgd update
            newWls[l] = self.Wls[l] - lr * dloss_dWl
            newW0ls[l] = self.W0ls[l] - lr * dloss_dW0l

        # replace the new weights in the weights attribute
        self.Wls, self.W0ls = newWls, newW0ls


    def train(self, X, Y, iter, lr = 0.05, show_result = False, show_plot = False, pause = False):
        
        if show_plot:
            xmin, xmax = min(X[0]), max(X[0])
            ymin, ymax = min(X[1]), max(X[1])
            ax = tidy_plot(xmin, xmax, ymin, ymax)
            
            def predictor(x, y):
                pguess = self.predict(np.array([[x, y]]).T)
                guess = 1 if pguess > 0.5 else -1
                return guess
        
        for it in range(iter):
            print(f'++++++ iteration number {it}')

            # pick a random x and y from dataset X and Y
            K = X.shape[-1]
            k = np.random.randint(0, K)
            x, y = X[:, k:k+1], Y[:, k:k+1]

            # feed forward and error back propagation on randomly choosed data
            self.forward(x, y)
            self.backward(x, y, lr)

            if show_result:
                print(f'++++++ after train number {it} +++++++')
                Wlst, W0lst = self.get_weights(silence=False)
                _, _ = self.get_Zls(), self.get_Als()

            if show_plot:

                labels = 2 * Y - 1
                plot_data(X, labels, ax, clear=True)

                Wlst, W0lst = self.get_weights()
                W_0, W0_0 = Wlst[0], W0lst[0]
                n = W_0.shape[-1]   # its in self.nls[0]
                for j in range(n):
                    th, th0 = W_0[:, j:j+1], W0_0[j:j+1, :]
                    plot_separator(ax, th, th0)
                    plot_nonlinear_separator(predictor, ax)
                
                plt.pause(0.01)
                if pause: input('next FF BP?')

        if show_plot: input('\nfinal result, done?')




    def predict(self, x):

        for l in range(self.L):
            Zl = self.Wls[l].T @ Al + self.W0ls[l] if l > 0 else self.Wls[l].T @ x + self.W0ls[l]
            Al = self.fls[l](Zl)
        return Al




# X = np.array([[2, 3]]).T
# Y = np.array([[1, 2]]).T

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[1, 0, 0, 1]])


# myNN = NN(2, (2, 2), (ReLU, ReLU), (dReLU, dReLU), squareloss, d_squareloss)
myNN = NN(2, (3, 1), (ReLU, sigmoid), (dReLU, dsigmoid), NLL, dNLL)


myNN.train(X, Y, 1000, 0.05, show_result=False, show_plot=True, pause=False)