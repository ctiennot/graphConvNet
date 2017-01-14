import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)


class GraphConv:
    """Convolution layer for Graphs, followed by RELU"""

    def __init__(self, input, K, F, L):
        """
        :param input: input from previous layer, size [None, K_, F_]
        :param K: number of Chebitchev coefficient
        :param F: number of filters
        :param L: Laplacian for the current graph
        """
        if F == 0 or K == 0 or K is None or F is None or L is None:
            # the layer doesn't apply filtering but can be used to just maxpool
            # or unpool
            self.K = None
            self.F = None
            self.L_tilde = None
            self.theta = None

            _, M_, F_ = input.get_shape()  # previous layer number of filters
            # and signal size
            F_, M_ = int(F_), int(M_)

            # the output is just the input
            self.output = input
            self.depth = F_  # same depth as input
            self.M = M_  # same signal size as input

        else:
            lambda_max = max(
                np.linalg.eig(L)[0])  # maximum eigenvalue of the Laplacian
            lambda_max = lambda_max.real  # avoid false imaginary part

            # Compute normalized Laplacian
            self.L_tilde = 2 * L / lambda_max - np.eye(L.shape[0])
            self.L_tilde = np.array(self.L_tilde, dtype=np.float32)
            self.M = self.L_tilde.shape[0]  # signal size

            self.K = K  # number of coef / Polynomial Degree
            self.F = F  # number of filters

            # Filter parameters
            self.theta = weight_variable([self.K, self.F])  # K x F
            self.b = bias_variable([1, 1, self.F])  # 1 x 1 x F

            # Perform symbolic operations
            _, _, F_ = input.get_shape()  # previous layer number of filters
            F_ = int(F_)

            X_bar = tf.reshape(input, [-1, self.M])  # NF_ x M
            X_bar = self._chebyshev_recursion(X_bar)  # NF_ x M x K

            y = self._apply_filtering(X_bar) # NF_ x M x F

            self.output = tf.nn.relu(y + self.b)  # NF_ x M x F

            self.output = tf.reshape(self.output, [-1, self.M, F_*self.F])

            # store depth (different for the number of filters for this layer)
            self.depth = F_*self.F

    def _chebyshev_recursion(self, X):
        # return N x M x K
        # X is N x M

        # Transform to Chebyshev basis
        x_t = tf.transpose(X)  # M x N
        x_t_exp = tf.expand_dims(x_t, 0)  # 1 x M x N

        def hstack(x_t_exp, x_t_1):
            x_t_1 = tf.expand_dims(x_t_1, 0)  # 1 x M x N
            return tf.concat(0, [x_t_exp, x_t_1])  # K x M x N

        # Initiliaze the recursion
        x_t_1 = tf.matmul(self.L_tilde, x_t)
        x_t_exp = hstack(x_t_exp, x_t_1)

        # add all Chebyshev \bar x
        for k in range(2, self.K):
            xt2 = 2 * tf.matmul(self.L_tilde, x_t_1) - x_t  # M x N
            x_t_exp = hstack(x_t_exp, xt2)
            x_t, x_t_1 = x_t_1, xt2
        x_t_exp = tf.transpose(x_t_exp)  # N x M x K

        return x_t_exp

    def _apply_filtering(self, X_bar):

        xt = tf.reshape(X_bar, [-1, self.K])  # NM x K
        # Apply the filters (by block)
        y = tf.matmul(xt, self.theta)  # NM x F
        y = tf.reshape(y, [-1, self.M, self.F])  # N x M x F

        return y

    def max_pool(self):
        """ Perform 1d-max_pooling after the convlayer"""
        augmented = tf.expand_dims(self.output, 0)
        pooled = tf.nn.max_pool(augmented, ksize=[1, 1, 2, 1],
                                  strides=[1, 1, 2, 1], padding='SAME')
        pooled = tf.squeeze(pooled, [0])
        self.output = pooled

        # update signal size
        self.M /= 2

    def un_pool(self):
        """
        :return: Unpooling on current layer output and so double the 1-d signal
        size by duplicating all nodes
        """
        self.output = tf.reshape(self.output,
                         [-1, self.M, 1, self.depth])  # add 1 dimension
        self.output = tf.tile(self.output, [1, 1, 2, 1])  # duplicate signal
        # column
        self.output = tf.reshape(self.output, [-1, 2 * self.M, self.depth])

        # update signal size
        self.M *= 2


class Dense:

    def __init__(self, input, u):
        """
        :param input:
        :param u: number of units
        """
        if len(input.get_shape())>2:
            # from conv to dense
            _, M, F = input.get_shape()
            M, F = int(M), int(F)

            self.W = weight_variable([M * F, u])
            self.b = bias_variable([u])

            y = tf.reshape(input, [-1, M * F])

            self.output = tf.matmul(y, self.W) + self.b

        elif len(input.get_shape())== 2:
            # from dense to dense
            _, M = input.get_shape()
            M = int(M)

            self.W = weight_variable([M, u])
            self.b = bias_variable([u])

            y = tf.reshape(input, [-1, M])

            self.output = tf.matmul(y, self.W) + self.b

    def relu(self):
        self.output = tf.nn.relu(self.output)

    def sofmax(self):
        self.output = tf.nn.softmax(self.output)