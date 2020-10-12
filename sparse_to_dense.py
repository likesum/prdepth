#!/usr/bin/env python3

import os
import argparse

import numpy as np
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
elif opts.nsparse == 50:
    GAMMA, NUM_GD_STEPS = 0.2, 1
elif opts.nsparse == 100:
    GAMMA, NUM_GD_STEPS = 0.3, 1
elif opts.nsparse == 200:
    GAMMA, NUM_GD_STEPS = 0.2, 2

#########################################################################

depth_sampler = sampler.Sampler(nsamples=100, read_gt=True)

optimizer = Optimizer(depth_sampler)

sess = tf.Session()
depth_sampler.load_model(sess)

#########################################################################
#### Main Loop
flist = [i.strip('\n') for i in open(TLIST).readlines()]
depths, preds = [], []

for filename in flist:
    # Load sparse input.
    sparse_depth = ut.read_depth(filename + '_sparseN%d.png' % opts.nsparse)

    # Run VAE to sample patch-wise predictions.
    depth_sampler.sample_predictions(filename, sess)

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
    pred[sparse_depth>0] = sparse_depth[sparse_depth>0]
    preds.append(pred)

    depth = sess.run(depth_sampler.image_depth).squeeze().astype(np.float64)
    depths.append(depth)

    if save_dir is not None:
        nm = os.path.join(save_dir, os.path.basename(filename))
        min_depth = np.maximum(0.01, np.min(depth))
        max_depth = np.minimum(1., np.max(depth))
        ut.save_color_depth(nm + '_gt.png', depth, min_depth, max_depth)
        ut.save_color_depth(nm + '_s2d.png', pred, min_depth, max_depth)

metrics = metric.get_metrics(depths, preds, projection_mask=True)
for k in metric.METRIC_NAMES:
    print("%s: %.3f" % (k, metrics[k]))
