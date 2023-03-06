
import numpy as np


# 1) Margin
# =========

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


def margin_of_points(datapoints, labels, th, th_0):
    '''
    Returns margin of datapoints with respect to separator th, th_0

    Parameters:
        datapoints (dxn array): n d-dimensional datapoint
        labels (1xn array): entries from {-1, +1}
        th (dx1 array)
        th_0 (1x1 array, or a scalar)

    Returns:
        (1xn array of) margins
    '''
    return labels * distance_to_separator(datapoints, th, th_0)


def sum_margin_score(th, th0, datapoints, labels):
    '''
    Returns sum of margins of datapoints (dxn array) with their labels (1xn array)
        with respect to separator with parameters th (dx1 array) and th0 (1x1 array or float)
        as a score for the separator on data.
    '''
    return np.sum(margin_of_points(datapoints, labels, th, th0))


def min_margin_score(th, th0, datapoints, labels):
    '''
    Return minimum of margins of datapoints (dxn array) with their labels (1xn array)
        with respect to separator with parameters th (dx1 array) and th0 (1x1 array or float)
        as a score for the separator on data.
    '''
    return np.min(margin_of_points(datapoints, labels, th, th0))


def max_margin_score(th, th0, datapoints, labels):
    '''
    Return maximum of margins of datapoints (dxn array) with their labels (1xn array)
        with respect to separator with parameters th (dx1 array) and th0 (1x1 array or float)
        as a score for the separator on data.
    '''
    return np.max(margin_of_points(datapoints, labels, th, th0))


data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])

red_th = np.array([[1, 0]]).T
red_th0 = -2.5

blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5

# print('red separator')
# print('sum margin score', sum_margin_score(red_th, red_th0, data, labels))
# print('min margin score', min_margin_score(red_th, red_th0, data, labels))
# print('max margin score', max_margin_score(red_th, red_th0, data, labels))

# print('blue separator')
# print('sum margin score', sum_margin_score(blue_th, blue_th0, data, labels))
# print('min margin score', min_margin_score(blue_th, blue_th0, data, labels))
# print('max margin score', max_margin_score(blue_th, blue_th0, data, labels))



# 4) Simply inseparable
# =====================

def hinge_loss(datapoints, labels, th, th0, ref):
    res = 1 - margin_of_points(datapoints, labels, th, th0) / ref
    res[res < 0] = 0
    return res


data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4
ref = 2**0.5 / 2

# print('hinge losses', hinge_loss(data, labels, th, th0, ref))



# 6.1) Gradient descent
# =====================

def f1(x):
    # x: one dimensional array vector
    return float((2 * x + 3)**2)

def df1(x):
    # x: one dimensional column vector
    return 2 * 2 * (2 * x + 3)

def f2(v):
    # v: 2 dimensional array vector
    x = float(v[0])
    y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def f2(v):
    # v: 2 dimensional array vector
    x = float(v[0])
    y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])

def cv(value_list):
    return np.array([value_list]).T



def gd(f, df, x0, step_size_fn, max_iter):
    '''
    Perform the gradient descent process by taking objective function f,
        gradient of the objective as df, initial parameter as x0,
        step size function which gives the step sizes of each step in gradient descent,
        and max iteration of the gradient descent process.

    f and df just take the parameter x which should be in the same shape of x0.

    This function returns a tuple of
        (final parameter, objective values history, parameters history)
    '''
    x = x0.copy()
    xs = [x0]
    fs = [f(x0)]
    for i in range(max_iter):
        x = x - step_size_fn(i) * df(x)
        xs.append(x)
        fs.append(f(x))
    return x, fs, xs


x, fs, xs = gd(f1, df1, cv([0.]), lambda i: 0.1, 1000)
print('final x', x)
print('size of fs', len(fs))



# 6.2) Numerical gradient
# =======================

def num_grad(f, delta=0.001):
    '''
    Wrapper function for numerical gradient calculator of function f,
        wrapped by the same function f, and delta as the telorance in approximating the gradient.

    f just takes a column array x as a whole parameter
        and returns a scalar.
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



# 7.1) Calculating the SVM objective
# ==================================

def hinge(v):
    # v is a 1xn vector
    return np.where(1-v > 0, 1-v, 0)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(y * (th.T @ x + th0))

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    return np.mean(hinge_loss(x, y, th, th0)) + lam * np.sum(th*th)



# 7.2) Calculating the SVM gradient
# =================================

# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    return np.where(1 - v > 0, -1, 0)

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    # result is a dxn vector which the column 'i' is the gradient of datapoint x[:,i]
    # chain rule is used here to calculate gradient of hinge loss with respect to th
    # v = y * (th.T @ x + th0) plays the intermediate variable role
    return y * x * d_hinge(y * (th.T @ x + th0))

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    # gradient of hinge loss with respect to th or th0 have different forms
    return y * d_hinge(y * (th.T @ x + th0))

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x, y, th, th0), axis=-1, keepdims=True) + 2*lam*th
    # first term above is the derivative of the training error with respect to th

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x, y, th, th0), axis=-1, keepdims=True)

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(x, y, th, th0, lam):
    return np.vstack([d_svm_obj_th(x, y, th, th0, lam), d_svm_obj_th0(x, y, th, th0, lam)])



# 7.3) Batch SVM minimize
# =======================

def batch_svm_min(data, labels, lam):
    '''
    Batch SVM minimizer which gives data and labels with a specified lambda,
        perform the gradient descent on the SVM objective
        (here the objective gradient is already defined by explicit calculation)
        and return the results as the
        (final parameter, list of objective values in the process, list of parameters in the process).
    '''
    def svm_min_step_size_fn(i):
        '''
        Step size function for the gradient descent.
        '''
        return 2/(i+1)**0.5
    d, n = data.shape
    # notice that we give the objective "function" (and also gradient objective function) 
    # to the gradient descent function
    # and objective function in the gd takes just a parameter as a whole
    # so we should redefine the objective function which just takes 1 parameter representing th and th0
    # and call the objective by th and th0 which have been taken from the input parameter
    f = lambda par: svm_obj(data, labels, par[:-1], par[-1:], lam)
    df = lambda par: svm_obj_grad(data, labels, par[:-1], par[-1:], lam)
    return gd(f, df, np.zeros((d+1, 1)), svm_min_step_size_fn, 10)



# 7.4) Numerical SVM objective
# ============================

def numerical_svm_min(data, labels, lam):

    def svm_min_step_size_fn(i):

        return 2 / (i+1)**0.5

    d, n = data.shape
    f = lambda par: svm_obj(data, labels, par[:-1], par[-1:], lam)
    df = num_grad(lambda par: svm_obj(data, labels, par[:-1], par[-1:], lam))
    return gd(f, df, np.zeros((d+1, 1)), svm_min_step_size_fn, 10)


def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y


# data, labels = super_simple_separable()
# lam = 0.0001
# x, fs, xs = numerical_svm_min(data, labels, lam)
# print('final parameter for numerical gradient descent', x)
# x, fs, xs = batch_svm_min(data, labels, lam)
# print('final parameter for explicit gradient descent', x)