import numpy as np
import tensorflow as tf

from prdepth import sampler
import prdepth.utils as ut
from prdepth.metric import PMASK


H, W = sampler.H, sampler.W
IH, IW = sampler.IH, sampler.IW
PSZ = sampler.PSZ
STRIDE = sampler.STRIDE
HNPS, WNPS = sampler.HNPS, sampler.WNPS


class DiverseOptimizer:
    ''' Optimizer class for yielding multiple diverse global estimations.

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

        # Build the diversity cost graph.
        self._build_diversity_cost_graph()

        # Graph for initializing the optimization, which needs to be run again
        # when generating a new global estimation.
        patched_init = tf.reduce_mean(self.patched_samples, axis=0)
        image_init = self.PO.group_patches(patched_init)
        self._optimization_init_op = tf.group([
            tf.assign(self.patched_before, patched_init).op,
            tf.assign(self.image_current, image_init).op])

        # Weight for diversity cost, which is slowly increased during the
        # optimization.
        self._lmd_ph = tf.placeholder(shape=[], dtype=tf.float32)

        # Select the sample with the min cost, which is the distance to the
        # (patch of) global prediction minus the diversity distance w.r.t
        # all previous global estimations.
        patched_current = self.PO.extract_patches(self.image_current)
        distance = ut.mean_diff(
            patched_current[None], self.patched_samples, axis=-1)
        diversity_cost = - self._diversity_distance / \
            (tf.cast(self._estimate_counter, tf.float32) + 1e-6)
        cost = tf.cond(
            self._estimate_counter > 0,
            lambda: distance + self._lmd_ph * diversity_cost,
            lambda: distance)
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

    def _build_diversity_cost_graph(self):
        ''' Build the graph for diversity cost.

        Including creating TF variables, graph for initializing and updating
        the diversity cost.
        '''

        # Variables for diversity distance.
        self._diversity_distance = tf.Variable(
            tf.zeros([self.nsamples, 1, HNPS, WNPS], dtype=tf.float32))
        self._estimate_counter = tf.Variable(tf.zeros([], dtype=tf.int32))

        # Graph for initializing the diverse estimates solver, which only needs
        # to be run once for all diverse global estimations.
        self._one_shot_init_op = tf.group([
            self._diversity_distance.initializer,
            self._estimate_counter.initializer])

        # Graph for updating the diversity cost after a global estimation is
        # returned.
        patched_current = self.PO.extract_patches(self.image_current)
        diversity_distance = ut.mean_diff(
            patched_current[None], self.patched_samples, axis=-1)
        self._update_diversity_op = tf.group([
            tf.assign_add(self._diversity_distance, diversity_distance).op,
            tf.assign_add(self._estimate_counter, 1).op])

    def initialize_diversity_cost(self, sess):
        ''' Initialize the diversity cost, which only needs to be run once for
        all diverse global estimations.
        '''
        sess.run(self._one_shot_init_op)

    def initialize_optimization(self, sess):
        ''' Initialize the optimization, which needs to be run again when 
        generating a new global estimation.
        '''
        sess.run(self._optimization_init_op)

    def update_global_estimation(self, sess):
        ''' Update the global depth estimation, which is simply the 
        overlap-average of the selected per-patch sample.

        Returns:
            Updated global estimation of the original resolution.
        '''
        return sess.run(self.resized_current)

    def update_sample_selection(self, lmd, sess):
        ''' Update sample selection using the current global estimation and the
        diversity cost to all previous global estimations.

        Args:
            lmd: weight for diversity cost.
            sess: TF session.
        Returns:
            Averaged squared difference of the current estimation and the 
            previous estimation.
        '''
        diff, _ = sess.run(
            [self._diff, self._sample_selection_op],
            feed_dict={self._lmd_ph: lmd})
        return diff

    def update_diversity_cost(self, sess):
        ''' Update the diversity cost after a global estimation is returned. '''
        sess.run(self._update_diversity_op)


class AnnotationOptimizer(DiverseOptimizer):
    ''' Optimizer class for user-interative estimation.

    For each estimation, the user annotates the the erroneous region. This user
    input is used to return a different global estimation.

    Optimizations are done in the DORN output resolution, which is lower than 
    the original image. The optimized global prediction is upsampled to the 
    original resolution.
    '''

    def __init__(self, depth_sampler, region_size):
        super().__init__(depth_sampler)
        self.region_size = region_size
        self._build_user_simulation_graph()

    def _build_diversity_cost_graph(self):
        ''' Build the graph for diversity cost in the user annotated erroneous
        region.

        Including creating TF variables, graph for initializing and updating
        the diversity cost.
        '''
        self._diversity_distance = tf.Variable(
            tf.zeros([self.nsamples, 1, HNPS, WNPS], dtype=tf.float32))
        self._estimate_counter = tf.Variable(tf.zeros([], dtype=tf.int32))

        # Graph for initializing the diverse estimates solver, which only needs
        # to be run once for all diverse global estimations.
        self._one_shot_init_op = tf.group([
            self._diversity_distance.initializer,
            self._estimate_counter.initializer])

        # Graph for updating the diversity cost after a global estimation is
        # returned and the user specify the region with the highest error.
        # The user input is given as a mask, in which a contiguous square is
        # all True indicating the erroneous region.
        # The additional cost in this case is the difference with the previous
        # estimation in this user marked region.
        self._erroneous_mask_ph = tf.placeholder(shape=[H, W], dtype=tf.bool)
        erroneous_mask = tf.cast(
            self._erroneous_mask_ph[None, :, :, None], tf.float32)
        image_mask = tf.image.resize_images(
            erroneous_mask, [IH, IW], align_corners=True)
        patched_mask = self.PO.extract_patches(image_mask)
        patched_current = self.PO.extract_patches(self.image_current)
        diversity_distance = tf.squared_difference(
            patched_current[None], self.patched_samples)
        diversity_distance = tf.reduce_mean(
            diversity_distance * patched_mask[None], axis=-1)
        self._update_diversity_op = tf.group([
            tf.assign_add(self._diversity_distance, diversity_distance).op,
            tf.assign_add(self._estimate_counter, 1).op])

    def _build_user_simulation_graph(self):
        ''' Build graph to simulate user annotation of erroneous regions. '''

        self._gt_depth_ph = tf.placeholder(shape=[H, W], dtype=tf.float32)
        # Variable to save which pixels have been previous annotated as
        # erroneous.
        self._marked_region = tf.Variable(
            tf.zeros([1, H, W, 1]), dtype=tf.float32)

        region_size = self.region_size
        valid_mask = PMASK * tf.cast(self._gt_depth_ph > 0, tf.float32)
        error = tf.squared_difference(
            self.resized_current, self._gt_depth_ph[None, :, :, None])
        error = error * valid_mask[None, :, :, None]
        window_error = tf.nn.avg_pool(
            error, [1, region_size, region_size, 1], [1, 1, 1, 1], 'VALID')
        # Regions of which 50% pixels were not marked before.
        window_marked = tf.nn.avg_pool(
            self._marked_region, [1, region_size, region_size, 1],
            [1, 1, 1, 1], 'VALID')
        window_marked = tf.where(
            window_marked < 0.5,
            tf.ones_like(window_marked),
            tf.zeros_like(window_marked))
        window_error = window_error * window_marked

        region_indices = tf.argmax(
            tf.reshape(window_error, [-1]), output_type=tf.int32)
        region_indices = tf.unravel_index(
            region_indices, tf.shape(window_error)[1:3])
        y, x = region_indices[0], region_indices[1]

        updated_marked_op = tf.assign(
            self._marked_region[0, y:y + region_size, x:x + region_size, 0],
            tf.ones([region_size, region_size])).op
        with tf.control_dependencies([updated_marked_op]):
            self._erroneous_region_loc = (tf.identity(y), tf.identity(x))

    def initialize_diversity_cost(self, sess):
        ''' Initialize the diversity cost, which only needs to be run once for
        all diverse global estimations.
        '''
        sess.run([self._one_shot_init_op, self._marked_region.initializer])

    def simulate_user_annotation(self, gt_depth, sess):
        ''' Simulates the user annotation of erroneous region.

        Simulated by selecting the squared window (of size region_size) with 
        the highest error while 50% of the pixels in the window are not marked
        before.

        Args:
            gt_depth: ground truth depth map.
            sess: TF session.
        Returns:
            a numpy boolean array with True indicating the erroneous region.
        '''
        y, x = sess.run(
            self._erroneous_region_loc,
            feed_dict={self._gt_depth_ph: gt_depth})
        erroneous_mask = np.zeros(shape=[H, W], dtype=np.bool)
        erroneous_mask[y:y + self.region_size, x:x + self.region_size] = True
        return erroneous_mask

    def update_diversity_cost(self, erroneous_mask, sess):
        ''' Update the diversity cost after a global estimation is returned and
        the user marked the erroneous region.

        Args:
            erroneous_mask: a numpy boolean array with True indicating the 
                erroneous region..
            sess: TF session.
        '''
        sess.run(
            self._update_diversity_op,
            feed_dict={self._erroneous_mask_ph: erroneous_mask})


