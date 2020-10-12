import tensorflow as tf

from .net import Net


class DORN(Net):

    def conv(self, name, inp, ksz, dil=1, stride=1, trainable=True):
        out = super().conv(
            name, inp, ksz, stride=stride, dil=dil,
            bias=True, relu='relu', pad='SAME', trainable=trainable)
        return out

    def caffe_pool(self, inp, ksz, strides, ptype='avg'):
        ''' TF implemenation of the pooling layer in caffe.
        The original DORN was implemented in caffe, which has a different
        pooling layer with tensorflow.
        '''
        if ptype == 'avg':
            pool = tf.nn.avg_pool
        elif ptype == 'max':
            pool = tf.nn.max_pool

        h, w = inp.get_shape().as_list()[1:3]
        kh, kw = ksz[1:3]
        sh, sw = strides[1:3]
        assert (h - kh) % sh != 0 and (w - kw) % sw != 0

        vh, vw = (h - kh) // sh * sh + sh, (w - kw) // sw * sw + sw

        out1 = pool(inp[:, :vh, :vw, :], ksz, strides, 'VALID')
        out2 = pool(inp[:, :vh, vw:, :], [1, kh, w - vw, 1],
                    [1, sh, sw, 1], 'VALID')
        out3 = pool(inp[:, vh:, :vw, :], [1, h - vh, kw, 1],
                    [1, sh, sw, 1], 'VALID')
        out4 = pool(inp[:, vh:, vw:, :], [1, h - vh,
                                          w - vw, 1], [1, sh, sw, 1], 'VALID')

        out1 = tf.concat([out1, out2], axis=2)
        out2 = tf.concat([out3, out4], axis=2)

        out = tf.concat([out1, out2], axis=1)
        return out

    def scene_understand(self, feat):
        keep_prob = 1.0
        bsz, height, width, inch = feat.get_shape().as_list()

        # Full image encoder
        out1 = self.caffe_pool(feat, [1, 8, 8, 1], [1, 8, 8, 1])
        out1 = tf.nn.dropout(out1, keep_prob=keep_prob)
        out1 = tf.reshape(out1, [bsz, 1, 1, -1])
        out1 = self.conv('scene/full/fc', out1, [1, 512])
        out1 = self.conv('scene/full/conv1', out1, [1, 512])
        out1 = tf.tile(out1, [1, height, width, 1])

        # ASPP (atrous spatial pyramid pooling) 1
        out2 = self.conv('ASPP1/conv1', feat, [1, 512], dil=1)
        out2 = self.conv('ASPP1/conv2', out2, [1, 512], dil=1)

        # ASPP 2
        out3 = self.conv('ASPP2/conv1', feat, [3, 512], dil=4)
        out3 = self.conv('ASPP2/conv2', out3, [1, 512], dil=1)

        # ASPP 3
        out4 = self.conv('ASPP3/conv1', feat, [3, 512], dil=8)
        out4 = self.conv('ASPP3/conv2', out4, [1, 512], dil=1)

        # ASPP 4
        out5 = self.conv('ASPP4/conv1', feat, [3, 512], dil=12)
        out5 = self.conv('ASPP4/conv2', out5, [1, 512], dil=1)

        out = tf.concat([out1, out2, out3, out4, out5], axis=-1)
        return out

    def ordinal_regress(self, feat, nch=136):
        keep_prob = 1.0

        out = tf.nn.dropout(feat, keep_prob=keep_prob)
        out = self.conv('Regress/conv1', out, [1, 2048])
        out = tf.nn.dropout(out, keep_prob=keep_prob)
        out = self.conv('Regress/conv2', out, [1, 136])
        h, w = out.get_shape().as_list()[1:3]
        out = tf.image.resize_bilinear(
            out, [(h - 1) * 8 + 1, (w - 1) * 8 + 1], align_corners=True)
        return out

    def decode_ordinal(self, ordinal):
        N, H, W, C = ordinal.get_shape().as_list()
        ordinal = tf.reshape(ordinal, [N, H, W, C // 2, 2])
        decode_label = tf.argmax(ordinal, axis=-1, output_type=tf.int32)
        decode_label = tf.to_float(decode_label)
        decode_label = tf.reduce_sum(decode_label, axis=-1, keepdims=True)
        depth = (decode_label - 1.0) / 25.0 - 0.36
        depth = tf.exp(depth) / 10.0
        return depth
