#!/usr/bin/env python3

import os
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow as tf

from prdepth import sampler
from prdepth import metric
import prdepth.utils as ut
from prdepth.optimization.s2d_optimizer import S2DOptimizer as Optimizer


parser = argparse.ArgumentParser()
parser.add_argument(
    '--nsparse', default=100, type=int, help='number of sparse samples')
parser.add_argument(
    '--save_dir', default=None, help='save predictions to where')
opts = parser.parse_args()
save_dir = opts.save_dir

TLIST = 'data/test.txt'
MAXITER = 200
TOLERANCE = 1e-8

if opts.nsparse == 20:
    GAMMA, NUM_GD_STEPS = 0.2, 1
    GAUSSIAN_SIGMA = 65
    NMS_SIZE = int(1.5 * GAUSSIAN_SIGMA)
elif opts.nsparse == 50:
    GAMMA, NUM_GD_STEPS = 0.2, 1
    GAUSSIAN_SIGMA = 35
    NMS_SIZE = int(3.0 * GAUSSIAN_SIGMA)
elif opts.nsparse == 100:
    GAMMA, NUM_GD_STEPS = 0.3, 1
    GAUSSIAN_SIGMA = 25
    NMS_SIZE = int(2.5 * GAUSSIAN_SIGMA)
elif opts.nsparse == 200:
    GAMMA, NUM_GD_STEPS = 0.2, 2
    GAUSSIAN_SIGMA = 35
    NMS_SIZE = int(1.0 * GAUSSIAN_SIGMA)

# Create a gaussian kernel
y, x = np.indices((2 * NMS_SIZE + 1, 2 * NMS_SIZE + 1))
kernel = ((y - NMS_SIZE)**2 + (x - NMS_SIZE)**2) / (2 * GAUSSIAN_SIGMA**2)
kernel = np.exp(-kernel)

#########################################################################

depth_sampler = sampler.Sampler(nsamples=100, read_gt=True)
patched_samples = depth_sampler.patched_samples

# Graph for computing uncertainty of the sampler.
PO = ut.PatchOp(1, sampler.IH, sampler.IW, sampler.PSZ, sampler.STRIDE)
patched_mean = tf.reduce_mean(patched_samples, axis=0)
image_mean = PO.group_patches(patched_mean)
patched_mean = PO.extract_patches(image_mean)
patched_var = tf.reduce_mean((patched_samples - patched_mean[None])**2, axis=0)
image_var = PO.group_patches(patched_var)
image_uncertainty = tf.image.resize_images(
    tf.sqrt(image_var), [sampler.H, sampler.W], align_corners=True)

optimizer = Optimizer(depth_sampler)

sess = tf.Session()
depth_sampler.load_model(sess)

#########################################################################
# Main Loop
flist = [i.strip('\n') for i in open(TLIST).readlines()]
depths, preds = [], []

for filename in flist:
    # Run VAE to sample patch-wise predictions.
    depth_sampler.sample_predictions(filename, sess)
    depth = sess.run(depth_sampler.image_depth).squeeze()

    # Run adaptive sampling with the estimated uncertainty map to generate the
    # sparse depth map.
    variance = sess.run(image_uncertainty).squeeze() ** 2.
    variance = gaussian_filter(
        variance, GAUSSIAN_SIGMA, mode='constant', cval=0.)
    sparse_depth = np.zeros_like(depth)
    for i in range(opts.nsparse):
        assert np.any(variance != 0)
        # Next location to sample is the location with the maximum uncertainty.
        y, x = np.unravel_index(np.argmax(variance, axis=None), variance.shape)
        sparse_depth[y, x] = depth[y, x]

        # Update the uncertainty map with soft non-maximum-supression.
        # The idea is the uncertainty in the window (of size 2 * NMS_SIZE + 1) 
        # centered at the newly sampled location should be reduced. The closer 
        # to the center, the uncertainty is reduced more.
        top = np.maximum(0, y - NMS_SIZE)
        left = np.maximum(0, x - NMS_SIZE)
        bot = np.minimum(sampler.H, y + NMS_SIZE)
        right =  np.minimum(sampler.W, x + NMS_SIZE)
        nms_kernel = kernel[
            NMS_SIZE - (y - top):NMS_SIZE + (bot - y),
            NMS_SIZE - (x - left):NMS_SIZE + (right - x)]
        h = variance[y, x] * nms_kernel
        variance[top:bot, left:right] = variance[top:bot, left:right] - h
        variance = np.maximum(variance, 0.)

    optimizer.initialize(sess)

    for i in range(MAXITER):
        global_current = optimizer.update_global_estimation(
            sparse_depth, GAMMA, NUM_GD_STEPS, sess)
        diff = optimizer.update_sample_selection(global_current, sess)

        if diff < TOLERANCE:
            break
    pred = optimizer.update_global_estimation(
        sparse_depth, GAMMA, NUM_GD_STEPS, sess)

    pred = np.clip(pred.squeeze(), 0.01, 1.).astype(np.float64)
    pred[sparse_depth > 0] = sparse_depth[sparse_depth > 0]
    preds.append(pred)

    depth = depth.astype(np.float64)
    depths.append(depth)

    if save_dir is not None:
        nm = os.path.join(save_dir, os.path.basename(filename))
        min_depth = np.maximum(0.01, np.min(depth))
        max_depth = np.minimum(1., np.max(depth))
        ut.save_color_depth(nm + '_gt.png', depth, min_depth, max_depth)
        ut.save_color_depth(nm + '_guideds2d.png', pred, min_depth, max_depth)
        ut.save_depth(nm + '_guided_sparse%d.png'%opts.nsparse, sparse_depth)

metrics = metric.get_metrics(depths, preds, projection_mask=True)
for k in metric.METRIC_NAMES:
    print("%s: %.3f" % (k, metrics[k]))
