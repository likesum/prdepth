import tensorflow as tf
import numpy as np

from .net import Net


class ResNet101(Net):

    def conv(self, name, inp, ksz, dil=1, stride=1):
        # ResNet uses Conv + BN + ReLu with same padding.
        out = super().conv(
            name, inp, ksz, stride=stride, dil=dil,
            bias=False, relu=False, pad='SAME', trainable=False)
        return out

    def unit(self, name, inp, nch, dil=1, stride=1, shortcut=False):
        out = self.conv(name + '/conv1', inp, [1, nch])
        out = tf.nn.relu(self.batchnorm(name + '/conv1', out))
        out = self.conv(name + '/conv2', out, [3, nch], stride=stride, dil=dil)
        out = tf.nn.relu(self.batchnorm(name + '/conv2', out))
        out = self.conv(name + '/conv3', out, [1, nch * 4])
        out = self.batchnorm(name + '/conv3', out)

        if shortcut:
            inp = self.conv(name + '/shortcut', inp,
                            [1, nch * 4], stride=stride)
            inp = self.batchnorm(name + '/shortcut', inp)

        return tf.nn.relu(inp + out)

    def block(self, name, nunit, inp, nch, dil=1, stride=1):
        out = self.unit(name + '/unit1', inp, nch, dil=dil,
                        stride=stride, shortcut=True)

        for i in range(2, nunit + 1):
            out = self.unit(name + '/unit%d' % i, out, nch, dil=dil)
        return out

    def get_feature(self, inp):
        # RGB to BGR
        red, green, blue = tf.split(inp, 3, -1)
        out = tf.concat([blue, green, red], 3)

        # Image-Net Mean subtraction
        bgr_mean = [103.0626, 115.9029, 123.1516]
        bgr_mean = np.reshape(bgr_mean, [1, 1, 1, 3])
        out = out - bgr_mean

        # Block 1
        out = self.conv('block1/conv1', out, [3, 64], stride=2)
        out = tf.nn.relu(self.batchnorm('block1/conv1', out))
        out = self.conv('block1/conv2', out, [3, 64], stride=1)
        out = tf.nn.relu(self.batchnorm('block1/conv2', out))
        out = self.conv('block1/conv3', out, [3, 128], stride=1)
        out = tf.nn.relu(self.batchnorm('block1/conv3', out))
        out = tf.nn.max_pool(out, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # Block 2
        out = self.block('block2', 3, out, nch=64, dil=1, stride=1)

        # Block 3
        out = self.block('block3', 4, out, nch=128, dil=1, stride=2)

        # Block 4
        out = self.block('block4', 23, out, nch=256, dil=2, stride=1)

        # Block 5
        out = self.block('block5', 3, out, nch=512, dil=4, stride=1)

        return out

    def init_model(self, path, sess):
        mfn = np.load(path)
        ph = tf.placeholder(tf.float32)

        for n in self.weights.keys():
            sess.run(self.weights[n].assign(ph), feed_dict={ph: mfn[n]})
