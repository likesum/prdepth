import tensorflow as tf
import tensorflow_probability as tfp
Gaussian = tfp.distributions.MultivariateNormalDiag

from prdepth.net import VAE
from prdepth.net import DORN
from prdepth.net import ResNet101
import prdepth.utils as ut

# We use half of the training stride, for a denser patch-fication.
# Correspondingly, we need a dilation factor of 2 since we are using half
# of the training stride.
STRIDE = 4
DILATION = 2
PSZ = 33

LATENT_DIM = 128

SHIFT = (PSZ - 1) // STRIDE
FACTOR = 8 // STRIDE

H, W = 480, 640  # Original image size
IH, IW = 257, 353  # Input image size to DORN
OH, OW = (IH - 1) // 8 + 1, (IW - 1) // 8 + 1  # 33x45, output size of DORN
FH, FW = (OH - 1) * FACTOR + 1, (OW - 1) * \
    FACTOR + 1  # The size of feature tensor
HNPS, WNPS = FH - SHIFT, FW - SHIFT  # Number of patches

DORN_PRETRAINED = 'wts/DORN_NYUv2.npz'
VAE_PATH = 'wts/iter_150000.model.npz'


class Sampler:
    '''A sampler class for sampling patch-wise depth predictions given a RGB.

    For a given RGB image, the feature extractor will be run once to extract
    features. The prior net of the VAE is then run once to get the prior 
    distribution. We then sample from the prior distribution to get multiple 
    latent codes and run the generator multiple times to sample multiple depth 
    predictions (samples) for each patch.

    Generated samples can then be used for monocular depth estimation or
    optimization with additional Information.

    Attributes:
        image_depth: a [1, H, W, 1] tf variable of GT depth map.
        patched_samples: tf variable for patch-wise samples of shape 
            [nsamples, 1, y_num_patches, x_num_patches, patch_size**2].
        read_gt: if read the ground-truth depth for evaluation.
    '''

    def __init__(self, nsamples=100, read_gt=True):
        self.nsamples = nsamples
        self.read_gt = read_gt
        self.build_graph()

    def build_graph(self):
        print('Initializing computational graph for sampling...')

        self.feature = tf.Variable(
            tf.zeros([1, OH, OW, 2560], dtype=tf.float32), trainable=False)
        self.patched_samples = tf.Variable(
            tf.zeros([self.nsamples, 1, HNPS, WNPS, PSZ**2], dtype=tf.float32),
            trainable=False)
        self.image_depth = tf.Variable(
            tf.zeros([1, H, W, 1], dtype=tf.float32), trainable=False)

        self.filename = tf.placeholder(tf.string)

        self.ResNet = ResNet101.ResNet101()
        self.SceneNet = DORN.DORN()
        self.VAE = VAE.VAE(latent_dim=LATENT_DIM, dil=DILATION)

        # Number of samples generated per-iteration.
        self.nsamples_piter = 1

        # Graph for image and GT depth loading.
        image = tf.read_file(self.filename + '_i.png')
        image = tf.image.decode_png(image, channels=3, dtype=tf.uint8)
        image = tf.cast(tf.stack([image]), tf.float32)
        image.set_shape([1, H, W, 3])

        if self.read_gt:
            depth = tf.read_file(self.filename + '_f.png')
            depth = tf.image.decode_png(depth, channels=1, dtype=tf.uint16)
            depth = tf.cast(tf.stack([depth]), tf.float32) / (2**16 - 1.0)
            depth.set_shape([1, H, W, 1])

        # Downsample the original image.
        image = tf.image.resize_images(image, [IH, IW], align_corners=True)

        # Graph for feature extraction.
        feat = self.ResNet.get_feature(image)
        feat = self.SceneNet.scene_understand(feat)

        if self.read_gt:
            self.feature_op = tf.group([
                tf.assign(self.feature, feat).op,
                tf.assign(self.image_depth, depth).op
            ])
        else:
            self.feature_op = tf.assign(self.feature, feat).op

        # Graph for generating the distribution for latent codes.
        if OH != FH:
            feat = tf.image.resize_bilinear(
                self.feature, [FH, FW], align_corners=True)
            feat = tf.tile(feat, [self.nsamples_piter, 1, 1, 1])
        else:
            feat = tf.tile(self.feature, [self.nsamples_piter, 1, 1, 1])
        prior_mean, prior_log_sigma = self.VAE.prior_net(feat)
        prior = Gaussian(loc=prior_mean, scale_diag=tf.exp(prior_log_sigma))

        # TF placeholder for sampling iteration.
        self.iter_ph = tf.placeholder(shape=[], dtype=tf.int32)

        # Sample latent vector (n predictions for each image)
        prior_latent = prior.sample()
        pred = self.VAE.generate(feat, prior_latent)
        pred = tf.reshape(pred, [self.nsamples_piter, 1, HNPS, WNPS, PSZ**2])

        # pred_op
        self.pred_op = tf.assign(
            self.patched_samples[
                self.iter_ph * self.nsamples_piter:(self.iter_ph + 1) * self.nsamples_piter],
            pred).op

        print('Done.')

    def load_model(self, sess):
        # Load saved weights for DORN
        DORN_weights = {**self.ResNet.weights, **self.SceneNet.weights}
        print("Loading DORN from " + DORN_PRETRAINED)
        ut.load_net(DORN_PRETRAINED, DORN_weights, sess)
        print("Done!")

        # Load VAE
        print("Loading VAE from " + VAE_PATH)
        ut.load_net(VAE_PATH, self.VAE.weights, sess)
        print("Done!")

        if self.read_gt:
            sess.run([v.initializer for v in [self.feature,
                                              self.image_depth,
                                              self.patched_samples]])
        else:
            sess.run([v.initializer for v in [self.feature,
                                              self.patched_samples]])

    def get_variables(self):
        ''' Return the tf Variable for patched depth samples'''
        if self.read_gt:
            return self.image_depth, self.patched_samples
        else:
            return self.patched_samples

    def sample_predictions(self, filename, sess):
        '''Given a filename of RGB image, sample patched depth predictions.'''

        # Run feature extractor
        sess.run(self.feature_op, feed_dict={self.filename: filename})

        for vi in range(self.nsamples // self.nsamples_piter):
            sess.run(self.pred_op, feed_dict={self.iter_ph: vi})
