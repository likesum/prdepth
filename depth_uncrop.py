#!/usr/bin/env python3

import os
import argparse

import numpy as np
import tensorflow as tf

from prdepth import sampler
from prdepth import metric
import prdepth.utils as ut
from prdepth.optimization.uncrop_optimizer import UncropOptimizer as Optimizer


parser = argparse.ArgumentParser()
parser.add_argument(
    '--height', default=120, type=int, help='height of the cropped depth')
parser.add_argument(
    '--width', default=160, type=int, help='width of the cropped depth')
parser.add_argument(
    '--save_dir', default=None, help='Save predictions to where')
opts = parser.parse_args()
save_dir = opts.save_dir

TLIST = 'data/test.txt'
MAXITER = 200
TOLERANCE = 1e-8
LMD = 150.

#########################################################################

depth_sampler = sampler.Sampler(nsamples=100, read_gt=True)

optimizer = Optimizer(depth_sampler, LMD)

sess = tf.Session()
depth_sampler.load_model(sess)

#########################################################################
# Main Loop
flist = [i.strip('\n') for i in open(TLIST).readlines()]
depths, preds, masks = [], [], []

for filename in flist:
    # Run VAE to sample patch-wise predictions.
    depth_sampler.sample_predictions(filename, sess)

    # Load cropped depth.
    depth = sess.run(depth_sampler.image_depth).squeeze()
    cropped_depth = ut.read_depth(
        filename + '_crop%dx%d.png' % (opts.height, opts.width))

    optimizer.initialize(sess)
    optimizer.compute_additional_cost(cropped_depth, sess)

    for i in range(MAXITER):
        global_current = optimizer.update_global_estimation(sess)
        diff = optimizer.update_sample_selection(sess)

        if diff < TOLERANCE:
            break
    pred = optimizer.update_global_estimation(sess)

    pred = np.clip(pred.squeeze(), 0.01, 1.).astype(np.float64)
    pred[cropped_depth > 0] = cropped_depth[cropped_depth > 0]
    preds.append(pred)

    depth = sess.run(depth_sampler.image_depth).squeeze().astype(np.float64)
    depths.append(depth)
    masks.append(cropped_depth == 0)

    if save_dir is not None:
        nm = os.path.join(save_dir, os.path.basename(filename))
        min_depth = np.maximum(0.01, np.min(depth))
        max_depth = np.minimum(1., np.max(depth))
        ut.save_color_depth(nm + '_gt.png', depth, min_depth, max_depth)
        ut.save_color_depth(nm + '_uncropped.png', pred, min_depth, max_depth)

# Metrics computed only on filled-in regions.
metrics = metric.get_metrics(depths, preds, projection_mask=True, masks=masks)
for k in metric.METRIC_NAMES:
    print("%s: %.3f" % (k, metrics[k]))
