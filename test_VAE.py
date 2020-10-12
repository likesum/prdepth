#!/usr/bin/env python3

import os
import argparse

import numpy as np
import tensorflow as tf

from prdepth import sampler
from prdepth import metric
import prdepth.utils as ut


parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_dir', default=None, help='Save predictions to where')
save_dir = parser.parse_args().save_dir

TLIST = 'data/test.txt'

#########################################################################
depth_sampler = sampler.Sampler(nsamples=100, read_gt=True)
image_depth, patched_samples = depth_sampler.get_variables()

#########################################################################
# Path for oracle prediction and average prediction

# An patch extractor to extract and group patches
PO = ut.PatchOp(1, sampler.IH, sampler.IW, sampler.PSZ, sampler.STRIDE)

# Get oracle distance
patched_depth = tf.image.resize_images(
    image_depth, [sampler.IH, sampler.IW], align_corners=True)
patched_depth = PO.extract_patches(patched_depth)
oracle_distance = tf.reduce_sum(
    tf.squared_difference(patched_depth, patched_samples), axis=-1)

# Get min and max indices
min_index = tf.argmin(oracle_distance, axis=0)
max_index = tf.argmax(oracle_distance, axis=0)
indices = tf.meshgrid(*[np.arange(i)
                        for i in min_index.get_shape().as_list()], indexing='ij')
min_indices = tf.stack([min_index] + indices, axis=-1)
max_indices = tf.stack([max_index] + indices, axis=-1)

# Get oracle prediction
patched_oracle = tf.gather_nd(patched_samples, min_indices)
image_oracle = PO.group_patches(patched_oracle)
image_oracle = tf.image.resize_images(
    image_oracle, [sampler.H, sampler.W], align_corners=True)

# Get mean prediction and the uncertainty map for the mean prediction, i.e. the 
# variance map.
patched_mean = tf.reduce_mean(patched_samples, axis=0)
image_mean = PO.group_patches(patched_mean)
patched_mean = PO.extract_patches(image_mean)
patched_var = tf.reduce_mean((patched_samples - patched_mean[None])**2, axis=0)
image_var = PO.group_patches(patched_var)
image_uncertainty = tf.image.resize_images(
    tf.sqrt(image_var), [sampler.H, sampler.W], align_corners=True)
image_mean = tf.image.resize_images(
    image_mean, [sampler.H, sampler.W], align_corners=True)

# Get adversary prediction
patched_adversary = tf.gather_nd(patched_samples, max_indices)
image_adversary = PO.group_patches(patched_adversary)
image_adversary = tf.image.resize_images(
    image_adversary, [sampler.H, sampler.W], align_corners=True)

#########################################################################
# Start TF session (respecting OMP_NUM_THREADS)
nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None:
    sess = tf.Session()
else:
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=int(nthr)))

depth_sampler.load_model(sess)

#########################################################################
# Main loop
# Input files
flist = [i.strip('\n') for i in open(TLIST).readlines()]
means, oracles, adversaries = [], [], []
depths = []

for filename in flist:

    depth_sampler.sample_predictions(filename, sess)

    # Get prediction
    oracle, mean, adversary, uncertainty, depth = sess.run(
        [image_oracle, image_mean, image_adversary, image_uncertainty, image_depth])

    oracle = np.clip(oracle.squeeze(), 0.01, 1.).astype(np.float64)
    mean = np.clip(mean.squeeze(), 0.01, 1.).astype(np.float64)
    adversary = np.clip(adversary.squeeze(), 0.01, 1.).astype(np.float64)
    uncertainty = uncertainty.squeeze()
    depth = depth.squeeze().astype(np.float64)

    oracles.append(oracle)
    means.append(mean)
    adversaries.append(adversary)
    depths.append(depth)

    if save_dir is not None:
        nm = os.path.join(save_dir, os.path.basename(filename))
        min_depth = np.maximum(0.01, np.min(depth))
        max_depth = np.minimum(1., np.max(depth))
        ut.save_color_depth(nm + '_gt.png', depth, min_depth, max_depth)
        ut.save_color_depth(nm + '_oracle.png', oracle, min_depth, max_depth)
        ut.save_color_depth(nm + '_mean.png', mean, min_depth, max_depth)
        ut.save_color_depth(
            nm + '_adversary.png', adversary, min_depth, max_depth)
        ut.save_color_depth(
            nm + '_uncertainty.png', uncertainty, 
            np.min(uncertainty), np.max(uncertainty), cmap=ut.HOT)


print(len(means))
oracle_metrics = metric.get_metrics(depths, oracles, projection_mask=True)
mean_metrics = metric.get_metrics(depths, means, projection_mask=True)
adversary_metrics = metric.get_metrics(
    depths, adversaries, projection_mask=True)

print("oracle     mean     adversary")
for k in metric.METRIC_NAMES:
    print("%s: %.3f,    %.3f,    %.3f" %
          (k, oracle_metrics[k], mean_metrics[k], adversary_metrics[k]))
