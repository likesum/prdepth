import tensorflow as tf
import numpy as np

from .net import Net


class VAE(Net):

    def __init__(self, dil=1, latent_dim=128):
        self.weights = {}
        self.trainable = {}
        self.dil = dil
        self.latent_dim = latent_dim

    def conv(self, name, inp, ksz, stride=1, bias=True, relu='relu', dil=1):
        out = super().conv(
            name, inp, ksz, stride=stride, dil=dil,
            bias=bias, relu=relu, pad='VALID', trainable=True)
        return out

    def conv_transpose(
            self, name, inp, ksz, outsp,
            stride=1, bias=True, pad='VALID', relu='relu'):
        out = super().conv_transpose(
            name, inp, ksz, outsp, stride=stride,
            bias=bias, relu=relu, pad=pad, trainable=True)
        return out

    def maxpool(inp, ksz, pad='VALID', stride=1):
        return tf.nn.pool(inp, [ksz, ksz], 'MAX', pad, [1, 1], [stride, stride])

    def prior_net(self, feat):
        bsz, hnps, wnps = feat.get_shape().as_list()[:3]
        out = self.conv('Prior_1', feat, [1, 1024], bias=True)
        out = self.conv('Prior_2', out, [1, 512], bias=True)
        out = self.conv('Prior_3', out, [3, 512], bias=True, dil=self.dil)
        out = self.conv('Prior_4', out, [3, 256], bias=True, dil=self.dil)

        out = self.conv('Prior_5', out, [1, 256], bias=True)
        out = self.conv('Prior_6', out, [1, 256], bias=True)
        out = self.conv(
            'Prior_7', out, [1, self.latent_dim * 2], bias=True, relu=False)
        return out[..., :self.latent_dim], out[..., self.latent_dim:]

    def posterior_net(self, feat, patch):
        bsz, hnps, wnps, psz = patch.get_shape().as_list()
        psz = int(np.sqrt(psz))
        out = tf.reshape(patch, [-1, psz, psz, 1])

        feat = self.conv(
            'Posterior_0', feat, [3, 1024], bias=True, dil=self.dil)
        feat = self.conv(
            'Posterior_1', feat, [3, 256], bias=True, dil=self.dil)

        out = self.conv(
            'Posterior_2', out, [3, 8], stride=2, bias=True)  # 16x16
        out = self.conv(
            'Posterior_3', out, [2, 16], stride=2, bias=True)  # 8x8
        out = self.conv(
            'Posterior_4', out, [2, 32], stride=2, bias=True)  # 4x4
        out = self.conv(
            'Posterior_5', out, [2, 64], stride=2, bias=True)  # 2x2

        out = tf.reshape(out, [bsz, hnps, wnps, -1])
        out = tf.concat([out, feat], axis=-1)

        out = self.conv('Posterior_6', out, [1, 1024], bias=True)
        out = self.conv('Posterior_7', out, [1, 512], bias=True)
        out = self.conv('Posterior_8', out, [1, 256], bias=True)
        out = self.conv(
            'Posterior_9', out, [1, self.latent_dim * 2], bias=True, relu=False)
        return out[..., :self.latent_dim], out[..., self.latent_dim:]

    def generate(self, feat, latent):
        bsz, hnps, wnps = feat.get_shape().as_list()[:3]
        out = self.conv('Gen_1', feat, [1, 1024], bias=True)
        out = self.conv('Gen_2', out, [1, 512], bias=True)
        out = self.conv('Gen_3', out, [3, 512], bias=True, dil=self.dil)
        out = self.conv('Gen_4', out, [3, 256], bias=True, dil=self.dil)

        out = tf.concat([out, latent], axis=-1)
        out = tf.reshape(out, [-1, 1, 1, out.get_shape().as_list()[-1]])
        hnps, wnps = hnps - 4 * self.dil, wnps - 4 * self.dil

        out = self.conv_transpose('Gen_8', out, [3, 256], 3, bias=True)
        out = self.conv_transpose('Gen_9', out, [3, 128], 5, bias=True)
        out = self.conv_transpose('Gen_10', out, [3, 64], 7, bias=True)
        out = tf.image.resize_images(out, [13, 13], align_corners=True)
        out = self.conv_transpose('Gen_11', out, [3, 32], 15, bias=True)
        out = self.conv_transpose('Gen_12', out, [3, 16], 17, bias=True)
        out = tf.image.resize_images(out, [33, 33], align_corners=True)

        out = self.conv('Gen_13', out, [1, 1], bias=True, relu='tanh')
        out = tf.reshape(out, [bsz, hnps, wnps, -1])

        return out
