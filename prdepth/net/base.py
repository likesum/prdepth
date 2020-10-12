import tensorflow as tf
import numpy as np

class VAE:
    def __init__(self, dil=1, latent_dim=128, test=False):
        self.weights = {}
        self.trainable = {}
        self.test = test
        self.dil = dil
        self.latent_dim = latent_dim


    def conv(self, name, inp, ksz, stride=1, bias=True, relu='relu', dil=1):
        nch = inp.get_shape().as_list()[-1]
        ksz = [ksz[0], ksz[0], nch, ksz[1]]

        wnm = name + "_w"
        if wnm in self.weights.keys():
            w = self.weights[wnm]
        else:
            sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[2]))
            w = tf.Variable(tf.random_uniform(ksz, minval=-sq, maxval=sq, dtype=tf.float32))
            self.weights[wnm] = w
            self.trainable[wnm] = w
        
        if dil == 1:
            out = tf.nn.conv2d(inp, w, [1,stride,stride,1], 'VALID')
        else:
            out = tf.nn.atrous_conv2d(inp, w, dil, 'VALID')
        
        if bias:
            bnm = name + "_b"
            if bnm in self.weights.keys():
                b = self.weights[bnm]
            else:
                b = tf.Variable(tf.constant(0,shape=[ksz[-1]],dtype=tf.float32))
                self.weights[bnm] = b
                self.trainable[bnm] = b
            out = out + b

        if relu == 'relu':
            out = tf.nn.relu(out)
        elif relu == 'leaky':
            out = tf.nn.leaky_relu(out, alpha=0.1)
        elif relu == 'tanh':
            out = tf.nn.tanh(out)

        return out


    def conv_transpose(self, name, inp, ksz, outsp, stride=1, bias=True, pad='VALID', relu='relu'):
        inpsp = inp.get_shape().as_list()
        bsz, nch = inpsp[0], inpsp[-1]
        ksz = [ksz[0],ksz[0],ksz[1],nch] # order of out_channels and in_channels

        wnm = name + "_w"
        if wnm in self.weights.keys():
            w = self.weights[wnm]
        else:
            sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[3]))
            w = tf.Variable(tf.random_uniform(ksz,minval=-sq,maxval=sq,dtype=tf.float32))
            self.weights[wnm] = w
            self.trainable[wnm] = w

        assert pad == 'SAME' or pad == 'VALID'
        outsp = [bsz,outsp,outsp,ksz[2]]

        out = tf.nn.conv2d_transpose(inp,w,outsp,[1,stride,stride,1],pad)

        if bias:
            bnm = name + "_b"
            if bnm in self.weights.keys():
                b = self.weights[bnm]
            else:
                b = tf.Variable(tf.constant(0,shape=[ksz[2]],dtype=tf.float32))
                self.weights[bnm] = b
                self.trainable[bnm] = b
            out = out+b

        if relu == 'relu':
            out = tf.nn.relu(out)
        elif relu == 'leaky':
            out = tf.nn.leaky_relu(out, alpha=0.1)
        elif relu == 'tanh':
            out = tf.nn.tanh(out)

        return out


    def maxpool(inp,ksz,pad='VALID',stride=1):
        return tf.nn.pool(inp,[ksz,ksz],'MAX',pad,[1,1],[stride,stride])
