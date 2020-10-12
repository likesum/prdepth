#!/usr/bin/env python3

import os
import argparse

import numpy as np
import tensorflow as tf

from prdepth import sampler
from prdepth import metric
import prdepth.utils as ut
from prdepth.optimization.interactive_optimizer import AnnotationOptimizer as Optimizer


parser = argparse.ArgumentParser()
parser.add_argument(
    '--n_estimation', default=10, type=int, help='number of estimations')
parser.add_argument(
    '--save_dir', default=None, help='save predictions to where')
opts = parser.parse_args()
save_dir = opts.save_dir

TLIST = 'data/test.txt'
MAXITER = 200
TOLERANCE = 1e-8
REGION_SIZE = 50
# Slowly increasing weight for diversity cost.
LMD = 10. * 2**(np.arange(50) / (50. - 1.) - 1.)

#########################################################################

depth_sampler = sampler.Sampler(nsamples=100, read_gt=True)

optimizer = Optimizer(depth_sampler, REGION_SIZE)

sess = tf.Session()
depth_sampler.load_model(sess)

#########################################################################
flist = [i.strip('\n') for i in open(TLIST).readlines()]
depths, preds = [], []

for filename in flist:
    # Run VAE to sample patch-wise predictions.
    depth_sampler.sample_predictions(filename, sess)
    depth = sess.run(depth_sampler.image_depth).squeeze().astype(np.float64)

    optimizer.initialize_diversity_cost(sess)

    diverse_estimations, rmses = [], []
    for m in range(opts.n_estimation):
        optimizer.initialize_optimization(sess)

        for i in range(MAXITER):
            lmd = LMD[i] if i < len(LMD) else LMD[-1]
            global_current = optimizer.update_global_estimation(sess)
            diff = optimizer.update_sample_selection(lmd, sess)

            if diff < TOLERANCE and i >= len(LMD):
                break
        pred = optimizer.update_global_estimation(sess)

        pred = np.clip(pred.squeeze(), 0.01, 1.).astype(np.float64)
        diverse_estimations.append(pred)

        rmse = metric.get_metrics(
            [depth], [pred], projection_mask=True, rmse_only=True)
        rmses.append(rmse)

        # Simulate the user annotation of the erroneous region in the returned
        # prediction.
        erroneous_mask = optimizer.simulate_user_annotation(depth, sess)
        optimizer.update_diversity_cost(erroneous_mask, sess)

    # Select the best from multiple diverse estimations.
    pred = diverse_estimations[np.argmin(rmses)]
    preds.append(pred)
    depths.append(depth)

    if save_dir is not None:
        print(np.argmin(rmses))
        nm = os.path.join(save_dir, os.path.basename(filename))
        min_depth = np.maximum(0.01, np.min(depth))
        max_depth = np.minimum(1., np.max(depth))
        ut.save_color_depth(nm + '_gt.png', depth, min_depth, max_depth)
        ut.save_color_depth(
            nm + '_interactive_best.png', pred, min_depth, max_depth)

metrics = metric.get_metrics(depths, preds, projection_mask=True)
for k in metric.METRIC_NAMES:
    print("%s: %.3f" % (k, metrics[k]))
