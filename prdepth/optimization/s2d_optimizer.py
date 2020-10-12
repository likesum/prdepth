import numpy as np
from scipy import ndimage as nd
import tensorflow as tf

from prdepth import sampler
import prdepth.utils as ut

import cv2

H, W = sampler.H, sampler.W
IH, IW = sampler.IH, sampler.IW
PSZ = sampler.PSZ
STRIDE = sampler.STRIDE
HNPS, WNPS = sampler.HNPS, sampler.WNPS


class S2DOptimizer:
    ''' Optimizer class for sparse-to-dense with random sampling.

    Optimizations are done in the DORN output resolution, which is lower than 
    the original image. The optimized global prediction is upsampled to the 
    original resolution.
    '''

    def __init__(self, depth_sampler):
        self.patched_samples = depth_sampler.patched_samples
        self.nsamples = depth_sampler.nsamples

        self.PO = ut.PatchOp(1, IH, IW, PSZ, STRIDE)

        # Variables for optimization
        # Global estimation (DORN resolution).
        self.image_current = tf.Variable(
            tf.zeros([1, IH, IW, 1], dtype=tf.float32))
        # Patches used to get image_current, i.e. if group average these 
        # patches, you would get image_current.
        self.patched_before = tf.Variable(
            tf.zeros([1, HNPS, WNPS, PSZ**2], dtype=tf.float32))
        # Global estimation (original resolution).
        self.resized_current = tf.image.resize_images(
            self.image_current, [H, W], align_corners=True)

        # Graph for initialization
        patched_init = tf.reduce_mean(self.patched_samples, axis=0)
        image_init = self.PO.group_patches(patched_init)
        self._init_op = tf.group([
            tf.assign(self.patched_before, patched_init).op,
            tf.assign(self.image_current, image_init).op])

        # Graph for updating sample selection (i.e., patched_before) based on
        # the updated global estimation, which is got by carrying out a few
        # number of gradient steps using the addtional global cost function.
        # In this sparse-to-dense application, the global estimation is updated
        # using sparse depth and Eq 9. & 10. in the paper.
        self._resized_updated_ph = tf.placeholder(
            shape=[H, W], dtype=tf.float32)
        image_current = tf.image.resize_images(
            self._resized_updated_ph[None, :, :, None],
            [IH, IW], align_corners=True)
        patched_current = self.PO.extract_patches(image_current)

        # Select the sample with the min distance to the (patch of) updated
        # global prediction.
        distance = ut.mean_diff(
            patched_current[None], self.patched_samples, axis=-1)
        min_index = tf.argmin(distance, axis=0)
        indices = tf.meshgrid(
            *[np.arange(i) for i in min_index.get_shape().as_list()], indexing='ij')
        min_indices = tf.stack([min_index] + indices, axis=-1)
        patched_best = tf.gather_nd(self.patched_samples, min_indices)
        image_best = self.PO.group_patches(patched_best)
        # Difference b/w the current prediction and the previous, used for
        # stopping the optimization.
        self._diff = ut.mean_diff(image_best, self.image_current)
        with tf.control_dependencies([self._diff]):
            self._sample_selection_op = tf.group([
                tf.assign(self.patched_before, patched_best).op,
                tf.assign(self.image_current, image_best).op])

    def initialize(self, sess):
        ''' Initialize the prediction. '''
        sess.run(self._init_op)

    def update_global_estimation(self, sparse_depth, gamma, num_gd_steps, sess):
        ''' Update the global depth estimation using sparse depth.

        By carrying out a few number of gradient steps using the addtional 
        global cost function (Eq 9. & 10. in the paper). The sampling operataion
        in this case is just sampling at the measured locations in the sparse
        depth map. The tranpose of the sampling operataion is nearest neighbor
        interpolation of the valid pixels in the sparse depth.

        Args:
            sparse_depth: a sparse depth map (numpy array).
            gamma: step size for gradient descent.
            num_gd_steps: number of gradient descent steps.
            sess: TF session.
        Returns:
            Updated global estimation of the original resolution.
        '''

        # A map of indices, of which each pixel is the indices of the closest
        # valid measurement on the sparse depth map to this pixel.
        # This is used for filling values for all pixels of the sparse depth map
        # using nearest neighbor.
        if not hasattr(self, '_edt_indices'):
            invalid = (sparse_depth == 0)
            self._edt_indices = tuple(nd.distance_transform_edt(
                invalid, return_distances=False, return_indices=True))

        global_current = sess.run(self.resized_current).squeeze()
        for i in range(num_gd_steps):
            diff = global_current - sparse_depth
            gradient = diff[self._edt_indices]
            global_current = global_current - gamma * gradient

        return global_current

    def update_sample_selection(self, global_current, sess):
        ''' Update sample selection using the current global estimation.

        Args:
            global_current: the current global depth estimation of the original
                resolution.
            sess: TF session.
        Returns:
            Averaged squared difference of the current estimation and the 
            previous estimation.
        '''
        diff, _ = sess.run(
            [self._diff, self._sample_selection_op],
            feed_dict={self._resized_updated_ph: global_current})
        return diff


class UpsamplingOptimizer(S2DOptimizer):
    def update_global_estimation(self, lowres_depth, gamma, num_gd_steps, sess):
        ''' Update the global depth estimation using low-resolution depth map.

        By carrying out a few number of gradient steps using the addtional 
        global cost function (Eq 9. & 10. in the paper). The sampling operataion
        in this case is bicubic downsampling. The tranpose of the sampling 
        operataion is bi-linear interpolation of the low-resolution depth.

        Args:
            lowres_depth: a low-resolution depth map (numpy array).
            gamma: step size for gradient descent.
            num_gd_steps: number of gradient descent steps.
            sess: TF session.
        Returns:
            Updated global estimation of the original resolution.
        '''

        global_current = sess.run(self.resized_current).squeeze()
        lh, lw = lowres_depth.shape
        for i in range(num_gd_steps):
            down_current = cv2.resize(
                global_current, (lw, lh), interpolation=cv2.INTER_CUBIC)
            diff = down_current - lowres_depth
            gradient = cv2.resize(diff, (W, H), interpolation=cv2.INTER_LINEAR)
            global_current = global_current - gamma * gradient

        return global_current