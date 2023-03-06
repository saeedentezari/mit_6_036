# my script of hw02

import matplotlib.pyplot as plt
import numpy as np


################################################################
# Utilities
################################################################

def norm(th):
    '''
    Norm of theta (dx1 array)
    '''
    return np.sqrt(np.sum(th*th))

def prediction(x, th, th_0):
    '''
    Prediction of label of datapoints x (dxn array)
        with respect to separator specified by parameters
        theta (dx1 array) and theta_0 (1x1 array).
    
    Returns (1xn array of) sign (int in {-1, 0, 1})
    '''
    return np.sign(np.dot(th.T, x) + th_0)

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
    if seed: np.random.seed(seed)
    np.random.shuffle(indices)
    # return shuffled dataset from shuffled indices
    return datapoints[:, indices], labels[:, indices]

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
        # draw zero axes
        ax.plot([0, 0], [ymin-epsy, ymax+epsy], c='k', lw=0.5)
        # or alternatively
        # ax.axvline(x=0, c='k', lw=0.5)
        ax.plot([xmin-epsx, xmax+epsx], [0, 0], c='k', lw=0.5)
        # ax.hline(y=0, c='k', lw=0.5)

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if equal_aspect: ax.set_aspect('equal')
    if grid: ax.grid(visible=True, ls=':')

    return ax

def plot_separator(ax, th, th_0):
    '''
    Plot separator in 2D with offset on given Axes object.

    Parameters:
        ax (matplotlib axes object): pre-prepared axes for plotting.
        th (2x1 array): theta
        th_0 (1x1 array): offset theta_0
    '''
    # keep track of plot limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # a list for putting in the interceptions
    pts = []
    th_0 = th_0[0,0]
    thx = th[0, 0]
    thy = th[1, 0]
    # find interception of separator with extension of horizontal borders
    if abs(thx) > 0:
        pts += [np.array([-(thy*y+th_0)/thx, y]) for y in (ymin, ymax)]
    # find interception of separator with extension of vertical borders
    if abs(thy) > 0:
        pts += [np.array([x, -(thx*x+th_0)/thy]) for x in (xmin, xmax)]

    # keep just interception points which are in the borders of the plot
    # and drop duplicates, in case of separator goes through the corner of the plot
    in_pts = []
    for pt in pts:
        if xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax:
            # check duplicate
            duplicate = False
            for p1 in in_pts:
                if np.max(np.abs(pt-p1)) == 0:
                    duplicate = True
            if not duplicate:
                in_pts.append(pt)

    # if separator is in the plot range
    if in_pts and len(in_pts) >= 2:

        # plot separator
        vpts = np.vstack(in_pts)
        ax.plot(vpts[:,0], vpts[:,1], c='black')

        # plot normal
        # midpoint of separator as the tail of the normal vector
        tailpt = (in_pts[0] + in_pts[1]) / 2
        # midpoint plus theta to get the head
        headpt = tailpt + th.T
        # stack tail and head, then plot normal
        nvpts = np.vstack((tailpt, headpt))
        ax.plot(nvpts[:,0], nvpts[:,1], c='black', ls=':')

        # try to keep limits from moving around
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    else:
        print(f'separator (th = {th}, th_0 = {th_0}) not in plot range')

def plot_data(datapoints, labels, ax = None, xmin = None, xmax = None, ymin = None, ymax = None,
                clear = False, **kwargs):
    '''
    Scatter plot data in 2D with labels as colors.

    Parameters:
        datapoints (2xn array): n 2d datapoints
        labels (1xn array): entries from set {-1, 0, +1}
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
    # if axis is not given, make  a new axis
    if ax is None:
        # if limits not given, take limits from datapoints
        if xmin == None: xmin = np.min(datapoints[0,:])
        if xmax == None: xmax = np.max(datapoints[0,:])
        if ymin == None: ymin = np.min(datapoints[1,:])
        if ymax == None: ymax = np.max(datapoints[1,:])
        # in case of max min equality, for example if we have just a datapoint
        # ... then extent the limits
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

def plot_pointer(ax, datapoints, labels, i, th, th_0):
    '''
    Plot a pointer around of ith point of dataset in 2D by drawing a circle arount it.
        This function consider the correctness of the current prediction
        of the point by green/red colors.

    Parameters:
        ax (matplotlib axes)
        datapoints (2xn array)
        labels (1xn array)
        i (int): ith point of the datapoints which is to be pointed.
            Should be in range(n).
        th (2x1 array)
        th_0 (1x1 array)
    '''

    # initialization
    d, n = datapoints.shape
    x_i = datapoints[:,i:i+1]
    y_i = labels[0,i:i+1]
    # color
    color = np.choose(prediction(x_i, th, th_0) == y_i, ['r', 'g'])[0,0]
    # scale
    xmin, xmax = ax.get_xlim()
    scale = (xmax - xmin) / 25
    # draw a circle around the specified datapoint
    draw_circle = plt.Circle((x_i[0,0], x_i[1,0]), radius=scale, fill=False, color=color)
    ax.add_artist(draw_circle)

################################################################
# Datasets
################################################################

big_data, big_data_labels = (np.array([[-2.04297103, -1.85361169, -2.65467827, -1.23013149, -0.31934782,
         1.33112127,  2.3297942 ,  1.47705445, -1.9733787 , -2.35476882,
        -4.97193554,  3.49851995,  4.00302943,  0.83369183,  0.41371989,
         4.37614714,  1.03536965,  1.2354608 , -0.7933465 , -3.85456759,
         3.22134658, -3.39787483, -1.31182253, -2.61363628, -1.14618119,
        -0.2174626 ,  1.32549116,  2.54520221,  0.31565661,  2.24648287,
        -3.33355258, -0.98689271, -0.24876636, -3.16008017,  1.22353111,
         4.77766994, -1.81670773, -3.58939471, -2.16268851,  2.88028351,
        -3.42297827, -2.74992813, -0.40293356, -3.45377267,  0.62400624,
        -0.35794507, -4.1648704 , -1.08734116,  0.22367444,  1.09067619,
         1.28738004,  2.07442478,  4.61951855,  4.47029706,  2.86510481,
         4.12532285,  0.48170777,  0.60089857,  4.50287515,  2.95549453,
         4.22791451, -1.28022286,  2.53126681,  2.41887277, -4.9921717 ,
         4.15022718,  0.49670572,  2.0268248 , -4.63475897, -4.20528418,
         1.77013481, -3.45389325,  1.0238472 , -1.2735185 ,  4.75384686,
         1.32622048, -0.13092625,  1.23457116, -1.69515197,  2.82027615,
        -1.01140935,  3.36451016,  4.43762708, -4.2679604 ,  4.76734154,
        -4.14496071, -4.38737405, -1.13214501, -2.89008477,  3.22986894,
         1.84103699, -3.91906092, -2.8867831 ,  2.31059245, -3.62773189,
        -4.58459406, -4.06343392, -3.10927054,  1.09152472,  2.99896855],
       [-2.1071566 , -3.06450052, -3.43898434,  0.71320285,  1.51214693,
         4.14295175,  4.73681233, -2.80366981,  1.56182223,  0.07061724,
        -0.92053415, -3.61953464,  0.39577344, -3.03202474, -4.90408303,
        -0.10239158, -1.35546287,  1.31372748, -1.97924525, -3.72545813,
         1.84834303, -0.13679709,  1.36748822, -2.92886952, -2.48367819,
        -0.0894489 , -2.99090327,  0.35494698,  0.94797491,  4.20393035,
        -3.14009852, -4.86292242,  3.2964068 , -0.9911453 ,  4.39465   ,
         3.64956975, -0.72225648, -0.15864119, -2.0340774 , -4.00758749,
         0.8627915 ,  3.73237594, -0.70011824,  1.07566463, -4.05063547,
        -3.98137177,  4.82410619,  2.5905222 ,  0.34188269, -1.44737803,
         3.27583966,  2.06616486, -4.43584161,  0.27795053,  4.37207651,
        -4.48564119,  0.7183541 ,  1.59374552, -0.13951634,  0.67825519,
        -4.02423434,  4.15893861, -1.52110278,  2.1320374 ,  3.31118893,
        -4.04072252,  2.41403912, -1.04635499,  3.39575642,  2.2189097 ,
         4.78827245,  1.19808069,  3.10299723,  0.18927394,  0.14437543,
        -4.17561642,  0.6060279 ,  0.22693751, -3.39593567,  1.14579319,
         3.65449494, -1.27240159,  0.73111639,  3.48806017,  2.48538719,
        -1.83892096,  1.42819622, -1.37538641,  3.4022984 ,  0.82757044,
        -3.81792516,  2.77707152, -1.49241173,  2.71063994, -3.33495679,
        -4.00845675,  0.719904  , -2.3257032 ,  1.65515972, -1.90859948]]), np.array([[-1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1., -1., -1.,  1.,
         1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1., -1.,  1., -1.,
        -1.,  1.,  1., -1., -1., -1.,  1., -1.,  1.,  1.,  1., -1., -1.,
         1., -1.,  1., -1.,  1., -1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,
        -1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1.,  1.,  1., -1., -1.,
         1.,  1.,  1.,  1., -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,
        -1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1., -1., -1.,  1., -1.,
        -1., -1.,  1., -1., -1., -1., -1.,  1.,  1.]]))

################## data for problem 1 and 2 ####################

def super_simple_seprable_through_origin():
    '''
    Returs three 2D datapoints (2x3 array) with their labels (1x3).
        Dataset is separable through origin.
    '''
    datapoints = np.array([[1, 0, -1.5], 
                        [-1, 1, -1]])
    labels = np.array([[1, -1, 1]])
    return datapoints, labels

################### data for problem 3 #########################

def problem3_data():
    datapoints = np.array(
    [
        [-3, -1, -1, 2, 1],
        [2, 1, -1, 2, -1],
    ]
)
    labels = np.array([[1, -1, -1, -1, -1]])
    return datapoints, labels

#################### data for problem 4.1 ######################

def problem4_1_data():
    datapoints = np.array(
        [
            (i, j, k) for i in range(2) for j in range(2) for k in range(2)
        ]
    ).T
    labels = np.array(list(map(lambda row: not 0 in row, datapoints.T)), ndmin=2)
    labels = 2 * labels - 1
    return datapoints, labels

####################### data generators ########################

def super_simple_seprable():
    '''
    Returns datapoints (2x4 array) with their labels (1x4 array).
        This dataset is seprable but not through origin.
    '''
    datapoints = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    labels = np.array([[1, -1, 1, -1]])
    return datapoints, labels

def xor():
    '''
    Returns datapoints (2x4 array) with their labels (1x4 array).
        Notice that this dataset is not linearly seprable.
    '''
    datapoints = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    labels = np.array([[1, 1, -1, -1]])
    return datapoints, labels

def gen_lin_seprable(npoints = 20, th = np.array([[1], [1]]), th_0 = np.array([[0.5]]), gamma = .01, R = 5.0):
    '''
    Randomly generates npoints which are linear seprable through
        separator of th and th_0, with minimum distance gamma to separator,
        and maximum distance R from origin.

    Parameters:
        npoints (int): number of points to generate.
        th (dx1 array)
        th_0 (1x1 array)
        gamma (float): minumum distance of points to the separator
        R (float): maximum distance of points to the origin

    Returns:
        datapoints (dxn array)
        labels (1xn array)
    '''
    # take dimension of points from th
    dim = th.shape[0]
    # an empty list for saving points
    datapoints = []
    # while we have not npoints
    while len(datapoints) < npoints:
        # draw a random point
        point = np.random.uniform(low=-R, high=R, size=(dim, 1))
        # if the point satisfies distance gamma from separator
        if abs(distance_to_separator(point, th, th_0)) >= gamma:
            # append the point to the datapoints
            datapoints.append(point)
    # create an array from datapoints list
    datapoints = np.hstack(datapoints)
    # determine labels of the points by prediction() function
    labels = prediction(datapoints, th, th_0)
    # return datapoints and labels
    return datapoints, labels

def wrap_gen_flipped(npoints = 20, pflip = 0.25,
                                th = np.array([[1], [1]]), th_0 = np.array([[0.5]]), gamma = .01, R = 5.0):
    '''
    Wrapper function that returns gen_flipped(),
        a function which returns a linear seprable dataset that its labels flipped randomly.
        Flipped dataset is not necessarily seprable.

    Params:
        Params of function gen_lin_seprable() and
        pflip (float): the probability of flipping each datapoint's label.

    Returns:
        gen_flipped (func) which takes number of datapoints,
            uses gen_lin_seprable() to make a dataset,
            then returns datapoints with its flipped labels.
    '''
    def gen_flipped(npoints = npoints):
        '''
        Flips labels of datapoints generated by gen_lin_seprable()
            with probability pflip.
        '''
        datapoints, labels = gen_lin_seprable(npoints, th, th_0, gamma, R)
        flip = np.random.uniform(low=0, high=1, size=(npoints,))
        for i in range(npoints):
            if flip[i] < pflip: labels[0,i] = -labels[0,i]
        return datapoints, labels
    return gen_flipped

def wrap_big_data_slicer():
    '''
    Returns a method that generates m points from dataset big_data, big_data_labels.
        Method will slice from current position to m points later.
        Initial current position is zero and if it overlaps n (number of samples in big_data),
        it would be returned back to zero.
    '''
    d, n = big_data.shape
    current = [0]
    def big_data_slicer(m):
        cur = current[0]
        data, labels = big_data[:,cur:cur+m], big_data_labels[:,cur:cur+m]
        current[0] += m
        if cur > n: current[0] = 0
        return data, labels
    
    return big_data_slicer

################################################################
# Perceptron
################################################################

def perceptron(datapoints, labels, params = {}, hook = None, count_mistakes = False):
    '''
    Perform perceptron algorithm for datapoints with labels,
        by going through datapoints T times.

    Parameters:
        datapoints (dxn array)
        labels (1xn array): values in {-1, 0, +1}
        params (dict, optional = {}): extra parameters to algorithm,
            such as T, the number of iterations through dataset.
        hook (func, optional = None): a function for plotting separator, if given.
        count_mistakes (bool, optional = False)

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
                if hook:
                    hook((th, th_0), i)
                th = th + y_i * x_i
                th_0 = th_0 + y_i
                # mistake counter
                nmistakes += 1

    if count_mistakes: print('Number of misatkes happened:', nmistakes)
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
                if hook: hook((th, th_0), i)
                th = th + y_i * x_i
                th_0 = th_0 + y_i
            # add up all th's in every iteration (correct or incorrect),
            # ... to get an average at the end
            thSum = thSum + th
            th_0Sum = th_0Sum + th_0
    thAvg = thSum / (T*n)
    th_0Avg = th_0Sum / (T*n)

    return thAvg, th_0Avg

###############################################################
# Tests
###############################################################

###################### test in action #########################

def test_linear_classifier(datafunc, learner, learner_params = {},
                            draw = True, refresh = True, pause = True):
    '''
    Test linear classifier by plotting in action.

    Parameters:
        datafunc (func): data generator function with all default arguments
        learner (func): could be perceptron or averaged perceptron
        learner_params (dict, optional = {}): parameters to pass to learner
        draw (bool, optional = True): if True, the hook function would be made for
            2D plotting each step (mistakes) of learning algorithm.
        refresh (bool, optional = True): if False, all the separator plotted will remain
            on the plot axes. Otherwise, one separator will be shown at a time.
        pause (bool, optional = True): if True, for going to next step of learning algorithm,
            user should press the enter key.
    '''

    # get datapoints and labels from data generator function
    datapoints, labels = datafunc()

    # if draw
    if draw:
        # plot data on an axes
        ax = plot_data(datapoints, labels)
        # define hook for use of plotting in learner
        def hook(params, i):
            th, th_0 = params
            # if we want to have just one separator on axes at a time
            # ... then we should clear axes before each plot_separator            
            if refresh: plot_data(datapoints, labels, ax, clear=True)
            # plot separator on axes
            plot_separator(ax, th, th_0)
            # plot a pointer around of datapoint with index i in dataset, which mistake happened on it
            plot_pointer(ax, datapoints, labels, i, th, th_0)
            # a default short pause
            plt.pause(0.05)
            # if pause is True, wait until user enters an input
            if pause: input('go?')

    # if not draw (else)
    else:
        # hook should be None
        hook = None

    # call learner with dataset and hook function created
    # get the learned parameters
    learned_th, learned_th_0 = learner(datapoints, labels, params=learner_params, hook=hook)
    # draw final result
    if draw:
        plot_data(datapoints, labels, ax, clear=True)
        plot_separator(ax, learned_th, learned_th_0)
        plt.pause(0.05)
        input('Final result. print?')

    # print score and learned params
    print('Final score on training dataset:', score(datapoints, labels, learned_th, learned_th_0))
    print('Learned params:\nth =', learned_th, '\nth_0 =', learned_th_0)

# test_linear_classifier(xor, perceptron, pause=True)

###################### test in expected #######################

expected_perceptron = [(np.array([[-24.0], [37.0]]), np.array([[-3.0]])), (np.array([[0.0], [-3.0]]), np.array([[0.0]]))]
expected_averaged_perceptron = [(np.array([[-22.1925], [34.06]]), np.array([[-2.1725]])), (np.array([[1.47], [-1.7275]]), np.array([[0.985]]))]
datagens = [super_simple_seprable, xor]

test_dict = {'perceptron': dict(zip(datagens, expected_perceptron)), 
                'averaged perceptron': dict(zip(datagens, expected_averaged_perceptron))}

def correct():
    print('Passed!\n')

def incorrect(result, expected):
    print('Test failed!')
    print('Your code output:', result)
    print('But expected:', expected)
    print('\n')

def test_perceptron():
    '''
    Check perceptron output for 100 iterations with its expected output.
        Data generators and their expected result is defined in above lines.
    '''
    for datagen, expected in test_dict['perceptron'].items():
        datapoints, labels = datagen()
        th, th_0 = perceptron(datapoints, labels, params={'T': 100})
        expected_th, expected_th_0 = expected
        print('-------- Test perceptron on ' + str(datagen.__name__) + ' data --------')
        if (th == expected_th).all() and (th_0 == expected_th_0).all():
            correct()
        else:
            incorrect('th = ' + str(th) + ', th_0 = ' + str(th_0), 'th = ' + str(expected_th) + ', th_0 = ' + str(expected_th))

def test_averaged_perceptron():
    '''
    Check averaged perceptron output for 100 iterations with its expected output.
        Data generators and their expected result is defined in above lines.
    '''
    for datagen, expected in test_dict['averaged perceptron'].items():
        datapoints, labels = datagen()
        th, th_0 = averaged_perceptron(datapoints, labels, params={'T': 100})
        expected_th, expected_th_0 = expected
        print('-------- Test averaged perceptron on ' + str(datagen.__name__) + ' data --------')
        if (th == expected_th).all() and (th_0 == expected_th_0).all():
            correct()
        else:
            incorrect('th = ' + str(th) + ', th_0 = ' + str(th_0), 'th = ' + str(expected_th) + ', th_0 = ' + str(expected_th))

###############################################################
# Evaluation
###############################################################

def eval_classifier(learner, data_train, labels_train, data_test, labels_test, draw_result = False):
    '''
    Perform classifier on training dataset and evaluate the score of outcome hypothesis on test dataset.

    Parameters:
        learner (func): learning algorithm, perceptron or averaged perceptron.
        data_train (dxntrain array)
        labels_train (1xntrain array)
        data_test (dxntest array)
        labels_test (1xntest array)
        draw_result (bool, optional = False): in 2D

    Returns:
        test_score (float) between 0 and 1
    '''
    # run learner with training dataset to get learned parameters
    learned_th, learned_th_0 = learner(data_train, labels_train)
    # evaluate the obtained separator on test dataset
    test_score = score(data_test, labels_test, learned_th, learned_th_0)
    # if draw train and test datasets
    if draw_result:
        ax = plot_data(data_train, labels_train)
        plot_data(data_test, labels_test, ax, marker='+')
        plot_separator(ax, learned_th, learned_th_0)
        print('test_score =', test_score)
        plt.pause(0.05)
        input('Done?')

    # return ratio of correctness of prediction with respect to obtained separator in test dataset
    return test_score

# # check and see classifier evaluation result by draw_result
# datapoints, labels = big_data, big_data_labels
# ncut = int(datapoints.shape[1] * 3 / 4)
# data_train, data_test = datapoints[:,:ncut], datapoints[:,ncut:]
# labels_train, labels_test = labels[:,:ncut], labels[:,ncut:]
# eval_classifier(perceptron, data_train, labels_train, data_test, labels_test, draw_result=True)

def eval_learning_alg(learner, data_gen, n_train, n_test, it=100, draw = False):
    '''
    Evaluate a learning algorithm 'it' times with dataset generated by data generator
        and returns the average score.

    Params:
        learner (func): learning algorithm, perceptron or averaged perceptron.
        data_gen (func): data generator function which
            takes the number of datapoints to generate
            and returns datapoints and its labels.
        n_train (int): number of points to generate training dataset.
        n_test (int): number of points to generate testing dataset.
        it (int, optional = 100): how many iterations to evaluate a learning algorithm.
        draw (bool, optional = False): if True, draw the result of every evaluation for 2D datapoints.

    Returns:
        average of evaluation scores (float between 0 to 1)
    '''
    scores = []
    # iterations over all evaluations to get average
    for i in range(it):
        # call data generator for making train and test datasets
        data_train, labels_train = data_gen(n_train)
        data_test, labels_test = data_gen(n_test)
        # evaluate specified datasets on learning algorithm
        score = eval_classifier(learner, data_train, labels_train, data_test, labels_test, draw_result=draw)
        # append each evaluation score to a list
        scores.append(score)
    
    # get average of evaluation scores and return
    return sum(scores) / it

# # test learning algorithm evaluation with flipped data generator
# it = 100
# print('Average evaluation score on {it} times ='.format(it=it),
#         eval_learning_alg(perceptron, wrap_gen_flipped(300, pflip=0.1), n_train=180, n_test=20, it=it, draw=False))

# # test learning alg eval with big_data slicer
# # compare this to cross-validation of big_data
# it = 10
# n = big_data.shape[1] // it
# n_test = n // 4
# n_train = n - n_test
# print('Average evaluation score =', eval_learning_alg(perceptron, wrap_big_data_slicer(), n_train, n_test, it, draw=True))

def xval_learning_alg(learner, datapoints, labels, k, draw = False):
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
        draw (bool, optional = False): if True, draw the result of every evaluation for 2D datapoints.

    Returns:
        average of test scores along the cross-validation process.
    '''
    # shuffle the dataset before start
    datapoints, labels = shuffle(datapoints, labels)
    # divide data into k chunks in a list
    datapoints_chunks = np.array_split(datapoints, k, axis=1)
    labels_chunks = np.array_split(labels, k, axis=1)
    # iteration k times for picking datasets and evaluations
    scores = []
    for i in range(k):
        # pick i'th chunk as test dataset
        datapoints_test, labels_test = datapoints_chunks[i], labels_chunks[i]
        # to get the training dataset, add the rest chunks together and concatenate them along axis=1 which was splitted
        datapoints_train = np.concatenate(datapoints_chunks[:i]+datapoints_chunks[i+1:], axis=1)
        labels_train = np.concatenate(labels_chunks[:i]+labels_chunks[i+1:], axis=1)
        # evaluate learner by the specified training and test datasets
        score = eval_classifier(learner, datapoints_train, labels_train, datapoints_test, labels_test, draw_result = draw)
        # append evaluation scores to a list
        scores.append(score)
    # get average of scores and return
    return sum(scores) / k


# # datapoints, labels = big_data, big_data_labels
# datapoints, labels = wrap_gen_flipped(npoints=200)()
# print('Cross-validation score =', xval_learning_alg(perceptron, datapoints, labels, k=10, draw=False))