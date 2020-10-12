import tensorflow as tf
import numpy as np


class Net:
    def __init__(self, lrmult=1.0, adam=False):
        self.weights = {}
        self.trainable = {}

    def conv(
            self, name, inp, ksz, stride=1, dil=1,
            bias=True, relu='relu', pad='SAME', trainable=True):
        nch = inp.get_shape().as_list()[-1]
        ksz = [ksz[0], ksz[0], nch, ksz[1]]
        assert (dil == 1 or stride == 1)

        wnm = name + "_w"
        if wnm in self.weights.keys():
            w = self.weights[wnm]
        else:
            sq = np.sqrt(3.0 / np.float32(ksz[0] * ksz[1] * ksz[2]))
            w = tf.Variable(
                tf.random_uniform(
                    ksz, minval=-sq, maxval=sq, dtype=tf.float32),
                trainable=trainable)
            self.weights[wnm] = w
            if trainable:
                self.trainable[wnm] = w

        if dil > 1:
            out = tf.nn.atrous_conv2d(inp, w, dil, pad)
        else:
            out = tf.nn.conv2d(inp, w, [1, stride, stride, 1], pad)

        if bias:
            bnm = name + "_b"
            if bnm in self.weights.keys():
                b = self.weights[bnm]
            else:
                b = tf.Variable(tf.constant(
                    0, shape=[ksz[-1]], dtype=tf.float32), trainable=trainable)
                self.weights[bnm] = b
                if trainable:
                    self.trainable[bnm] = b
            out = out + b

        if relu == 'relu':
            out = tf.nn.relu(out)
        elif relu == 'leaky':
            out = tf.nn.leaky_relu(out, alpha=0.1)
        elif relu == 'tanh':
            out = tf.nn.tanh(out)

        return out

    def conv_transpose(
            self, name, inp, ksz, outsp, stride=1,
            bias=True, relu='relu', pad='SAME', trainable=True):
        inpsp = inp.get_shape().as_list()
        bsz, nch = inpsp[0], inpsp[-1]
        # order of out_channels and in_channels
        ksz = [ksz[0], ksz[0], ksz[1], nch]

        wnm = name + "_w"
        if wnm in self.weights.keys():
            w = self.weights[wnm]
        else:
            sq = np.sqrt(3.0 / np.float32(ksz[0] * ksz[1] * ksz[3]))
            w = tf.Variable(
                tf.random_uniform(
                    ksz, minval=-sq, maxval=sq, dtype=tf.float32),
                trainable=trainable)
            self.weights[wnm] = w
            if trainable:
                self.trainable[wnm] = w

        assert pad == 'SAME' or pad == 'VALID'
        outsp = [bsz, outsp, outsp, ksz[2]]

        out = tf.nn.conv2d_transpose(
            inp, w, outsp, [1, stride, stride, 1], pad)

        if bias:
            bnm = name + "_b"
            if bnm in self.weights.keys():
                b = self.weights[bnm]
            else:
                b = tf.Variable(tf.constant(
                    0, shape=[ksz[2]], dtype=tf.float32), trainable=trainable)
                self.weights[bnm] = b
                if trainable:
                    self.trainable[bnm] = b
            out = out + b

        if relu == 'relu':
            out = tf.nn.relu(out)
        elif relu == 'leaky':
            out = tf.nn.leaky_relu(out, alpha=0.1)
        elif relu == 'tanh':
            out = tf.nn.tanh(out)

        return out

    def batchnorm(self, name, inp):
        if name + '_mean' not in self.weights.keys():
            size = inp.get_shape().as_list()[-1]
            mean = tf.Variable(tf.random_uniform(
                [size], dtype=tf.float32), collections=[], trainable=False)
            var = tf.Variable(tf.random_uniform(
                [size], dtype=tf.float32), collections=[], trainable=False)
            beta = tf.Variable(tf.random_uniform(
                [size], dtype=tf.float32), collections=[], trainable=False)
            gamma = tf.Variable(tf.random_uniform(
                [size], dtype=tf.float32), collections=[], trainable=False)

            self.weights[name + '_mean'] = mean
            self.weights[name + '_var'] = var
            self.weights[name + '_beta'] = beta
            self.weights[name + '_gamma'] = gamma

        else:
            mean = self.weights[name + '_mean']
            var = self.weights[name + '_var']
            beta = self.weights[name + '_beta']
            gamma = self.weights[name + '_gamma']

        return tf.nn.batch_normalization(inp, mean, var, beta, gamma, 1e-5)
