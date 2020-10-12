import numpy as np
import tensorflow as tf

from prdepth import sampler
import prdepth.utils as ut


H, W = sampler.H, sampler.W
IH, IW = sampler.IH, sampler.IW
PSZ = sampler.PSZ
STRIDE = sampler.STRIDE
HNPS, WNPS = sampler.HNPS, sampler.WNPS


class UncropOptimizer:
    ''' Optimizer class for uncropping.

    Optimizations are done in the DORN output resolution, which is lower than 
    the original image. The optimized global prediction is upsampled to the 
    original resolution.
    '''

    def __init__(self, depth_sampler, lmd):
        '''
        Args:
            depth_sampler: patch-wise depth sampler.
            lmd: lambda for the additional per-patch cost.
        '''
        self.patched_samples = depth_sampler.patched_samples
        self.nsamples = depth_sampler.nsamples
        self.lmd = lmd

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
        # the per-patch cost, which is computed using the available dense
        # measurements in a contiguous (but small) portion of the image .
        # This per-patch cost is computed only once on all samples at the start
        # of the optimization.
        self._patch_cost = tf.Variable(
            tf.zeros([self.nsamples, 1, HNPS, WNPS], dtype=tf.float32))
        self._cropped_ph = tf.placeholder(shape=[H, W], dtype=tf.float32)
        image_mask = tf.cast(self._cropped_ph > 0., tf.float32)
        image_mask = tf.image.resize_images(
            image_mask[None, :, :, None], [IH, IW], align_corners=True)
        image_cropped = tf.image.resize_images(
            self._cropped_ph[None, :, :, None], [IH, IW], align_corners=True)
        image_cropped = image_cropped / (image_mask + 1e-8)
        patched_cropped = self.PO.extract_patches(image_cropped)
        patched_mask = self.PO.extract_patches(image_mask)
        cost = (patched_cropped[None] - self.patched_samples)**2
        cost = tf.reduce_sum(cost * patched_mask[None], axis=-1)
        self._compute_cost_op = tf.assign(self._patch_cost, cost)

        # Select the sample with the min cost, which is the distance to the
        # (patch of) global prediction + the cost w.r.t small fov measurements.
        patched_current = self.PO.extract_patches(self.image_current)
        distance = ut.mean_diff(
            patched_current[None], self.patched_samples, axis=-1)
        cost = distance + self.lmd * self._patch_cost
        min_index = tf.argmin(cost, axis=0)
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

    def compute_additional_cost(self, cropped_depth, sess):
        ''' Compute the addtional per-patch cost w.r.t cropped depth map '''
        sess.run(
            self._compute_cost_op, feed_dict={self._cropped_ph: cropped_depth})

    def update_global_estimation(self, sess):
        ''' Update the global depth estimation, which is simply the 
        overlap-average of the selected per-patch sample.

        Returns:
            Updated global estimation of the original resolution.
        '''
        return sess.run(self.resized_current)

    def update_sample_selection(self, sess):
        ''' Update sample selection using the current global estimation and the
        per-patch addtional cost.

        Returns:
            Averaged squared difference of the current estimation and the 
            previous estimation.
        '''
        diff, _ = sess.run([self._diff, self._sample_selection_op])
        return diff
