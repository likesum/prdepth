#!/usr/bin/env python3

import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
Gaussian = tfp.distributions.MultivariateNormalDiag

from prdepth.dataset import Dataset
import prdepth.utils as ut
from prdepth.net import DORN
from prdepth.net import ResNet101
from prdepth.net import VAE


#########################################################################
# Training Hyper-Params

TLIST = 'data/NYUtrain.txt'
VLIST = 'data/NYUval.txt'

PSZ = 33
STRIDE = 8
LATENT_DIM = 128

SHIFT = (PSZ - 1) // STRIDE
FACTOR = 8 // STRIDE

H, W = 480, 640  # Original image size
IH, IW = 257, 353  # Input image size to DORN
OH, OW = (IH - 1) // 8 + 1, (IW - 1) // 8 + 1  # 33x45, output size of DORN
FH, FW = (OH - 1) * FACTOR + 1, (OW - 1) * \
    FACTOR + 1  # The size of feature tensor
HNPS, WNPS = FH - SHIFT, FW - SHIFT  # Number of patches

# Batch size for feature extractor (DORN) and VAE.
DORN_BSZ = 2
N_DORN_BATCH = 2
BSZ = DORN_BSZ * N_DORN_BATCH
ABSZ = BSZ * HNPS * WNPS

# Params for Adam
LR = 1e-4
BETA1 = 0.5
BETA2 = 0.9

# weight for KL divergence loss.
LMD = 1e-4

VALFREQ = 5e3
SAVEFREQ = 1e4
MAXITER = 1.5e5

# Run the sampler 10 times when val to generate 10 samples per-patch
VARIter = 10

WTS = 'wts'
DORN_PRETRAINED = 'wts/DORN_NYUv2.npz'

if not os.path.exists(WTS):
    os.makedirs(WTS)

# Check for saved weights & optimizer states
msave = ut.ckpter(WTS + '/iter_*.model.npz')
ssave = ut.ckpter(WTS + '/iter_*.state.npz')
ut.logopen(WTS + '/train.log')
niter = msave.iter

#########################################################################
# Feature extraction network is DORN (ResNet-101 based), which is pre-trained
# and fixed when training the VAE.

ResNet = ResNet101.ResNet101()
SceneNet = DORN.DORN()

#########################################################################
# Variables that are persistent across sessions.

gt = tf.Variable(tf.zeros([BSZ, H, W, 1], dtype=tf.float32), trainable=False)
patched_depth = tf.Variable(
    tf.zeros([BSZ, HNPS, WNPS, PSZ * PSZ], dtype=tf.float32), trainable=False)
patched_mask = tf.Variable(
    tf.zeros([BSZ, HNPS, WNPS, 1], dtype=tf.float32), trainable=False)

patched_pred = tf.Variable(tf.zeros(
    [VARIter, BSZ, HNPS, WNPS, PSZ * PSZ], dtype=tf.float32), trainable=False)
feature = tf.Variable(
    tf.zeros([BSZ, OH, OW, 2560], dtype=tf.float32), trainable=False)

#########################################################################
# Graph for feature extraction network, which is run multiple times to
# construct a batch of extracted features for monocular RGB images.
# Extracted features are saved to persistent variables, which can
# then be used for one training iteration in the VAE training graph/session.

biter = tf.placeholder(shape=[], dtype=tf.int32)

tset = Dataset(TLIST, DORN_BSZ, H, W, niter * N_DORN_BATCH,
               isval=False, aug=True, seed=0)
vset = Dataset(VLIST, DORN_BSZ, H, W, 0, isval=True, aug=False)
depth, image, swpT, swpV = tset.tvSwap(vset)

# Downsample the original image to the DORN input resolution.
image = tf.image.resize_images(image, [IH, IW], align_corners=True)

# ResNet
feat = ResNet.get_feature(image)

# Scene Understanding
feat = SceneNet.scene_understand(feat)
feature_op = tf.assign(
    feature[biter * DORN_BSZ:(biter + 1) * DORN_BSZ], feat).op

# GT
gt_op = tf.assign(gt[biter * DORN_BSZ:(biter + 1) * DORN_BSZ], depth).op

# Depth patches from downsampled depth map
down_depth = tf.image.resize_images(depth, [IH, IW], align_corners=True)
pdepth = tf.extract_image_patches(down_depth, [1, PSZ, PSZ, 1], [
                                  1, STRIDE, STRIDE, 1], [1, 1, 1, 1], 'VALID')
depth_op = tf.assign(
    patched_depth[biter * DORN_BSZ:(biter + 1) * DORN_BSZ], pdepth).op

# Masks
mask = tf.image.resize_images(
    tf.to_float(depth > 0), [IH, IW], align_corners=True)
mask = tf.extract_image_patches(
    mask, [1, PSZ, PSZ, 1], [1, STRIDE, STRIDE, 1], [1, 1, 1, 1], 'VALID')
mask = tf.to_float(tf.reduce_all(tf.equal(mask, 1.0), axis=-1, keepdims=True))
mask_op = tf.assign(patched_mask[biter * DORN_BSZ:(biter + 1) * DORN_BSZ], mask)

prepare_op = tf.group([feature_op, gt_op, depth_op, mask_op])


#########################################################################
# Graph for training VAE, which uses features from feature extractor
# The VAE contains three parts and is trained as follows:
#   prior net: 
#       input: features for each patch from the RGB image.
#       output: mean and log-sigma for prior distribution of the latent code.
#       trained by: minimizing kl divergence versus the posterior distribution.
#   posterior net:
#       input: features for each patch from the RGB image, depth for each patch.
#       output: mean and log-sigma for posterior distribution of the latent 
#               code.
#       trained by: minimizing the l1 loss between the GT depth and the 
#                   generated depth by using a sampled latent code from this
#                   posterior distribution.
#   generator:
#       input: features for each patch from the RGB image, latent code sampled
#              from either the prior distribution or the posterior distribution.
#       output: depth prediction for each patch.
#       trained by: minimizing the l1 loss between the GT depth and prediction.
#
# At inference time, posterior net will be discarded. We will run the feature
# extractor once to get features for each patch, and then run the prior net once
# to get the prior distribution. We then sample from the prior distribution 
# to get multiple latent codes and run the generator multiple times to sample
# multiple depth predictions (samples) for each patch.


if OH != FH:
    feature = tf.image.resize_bilinear(feature, [FH, FW], align_corners=True)

VAE_model = VAE.VAE(latent_dim=LATENT_DIM)

prior_mean, prior_log_sigma = VAE_model.prior_net(feature)
posterior_mean, posterior_log_sigma = VAE_model.posterior_net(
    feature, patched_depth)

prior = Gaussian(loc=prior_mean, scale_diag=tf.exp(prior_log_sigma))
posterior = Gaussian(loc=posterior_mean,
                     scale_diag=tf.exp(posterior_log_sigma))

posterior_latent = posterior.sample()
depth_pred = VAE_model.generate(feature, posterior_latent)

l1_loss = tf.reduce_mean(
    tf.abs(depth_pred - patched_depth), axis=-1, keepdims=True)
l1_loss = tf.reduce_sum(l1_loss * patched_mask)

# KL divergence
kl_loss = tfp.distributions.kl_divergence(posterior, prior)
kl_loss = tf.reduce_sum(kl_loss[..., None] * patched_mask)

loss = l1_loss + kl_loss * LMD

# Set up optimizer
lr = tf.placeholder(shape=[], dtype=tf.float32)
opt = tf.train.AdamOptimizer(lr, beta1=BETA1, beta2=BETA2)
tstep = opt.minimize(loss, var_list=list(VAE_model.weights.values()))

# Get loss to print
nvalids = tf.reduce_sum(patched_mask)
lvals = [loss / nvalids, l1_loss / nvalids, kl_loss / nvalids]
lnms = ['loss', 'L1', 'KL']
tnms = [l + '.t' for l in lnms]

# Save predictions from posterior net to compute average and min L2 loss later
l2_iter = tf.placeholder(shape=[], dtype=tf.int32)
l2_op = tf.assign(patched_pred[l2_iter], depth_pred).op


#########################################################################
# Graph to compute average and min L2 loss using samples generated with VAE.

# A patch extractor to extract and group patches.
PO = ut.PatchOp(BSZ, IH, IW, PSZ, STRIDE)

# Get oracle distance of all prediction samples to the GT depth patch.
oracle_distance = tf.reduce_sum(tf.squared_difference(
    patched_depth[tf.newaxis], patched_pred), axis=-1)

# Get min indices of the best prediction sample for each patch.
min_index = tf.argmin(oracle_distance, axis=0)
indices = tf.meshgrid(*[np.arange(i)
                        for i in min_index.get_shape().as_list()], indexing='ij')
min_indices = tf.stack([min_index] + indices, axis=-1)

# Get the best prediction sample for each patch.
patched_oracle = tf.gather_nd(patched_pred, min_indices)
image_oracle = PO.group_patches(patched_oracle)
image_oracle = tf.image.resize_images(image_oracle, [H, W], align_corners=True)
l2_mask = tf.to_float(gt > 0.)
min_l2 = (image_oracle - gt)**2
min_l2 = tf.reduce_sum(min_l2 * l2_mask)

# Get mean prediction for each patch.
patched_mean = tf.reduce_mean(patched_pred, axis=0)
image_mean = PO.group_patches(patched_mean)
image_mean = tf.image.resize_images(image_mean, [H, W], align_corners=True)
avg_l2 = (image_mean - gt)**2
avg_l2 = tf.reduce_sum(avg_l2 * l2_mask)

l2_nvalids = tf.reduce_sum(l2_mask)


#########################################################################
# Start TF session (respecting OMP_NUM_THREADS)
nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None:
    sess = tf.Session()
else:
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=int(nthr)))
sess.run(tf.global_variables_initializer())

# Load pre-trained weights for DORN
DORN_weights = {**ResNet.weights, **SceneNet.weights}
ut.mprint("Loading DORN from " + DORN_PRETRAINED)
ut.load_net(DORN_PRETRAINED, DORN_weights, sess)
print("Done!")

# Load saved weights for VAE if any
if niter > 0:
    mfn = WTS + "/iter_%06d.model.npz" % niter
    sfn = WTS + "/iter_%06d.state.npz" % niter

    ut.mprint("Restoring model from " + mfn)
    ut.load_net(mfn, VAE_model.weights, sess)
    ut.mprint("Restoring state from " + sfn)
    ut.load_adam(sfn, opt, VAE_model.weights, sess)
    ut.mprint("Done!")


#########################################################################
# Main Training loop

ut.stop = False
ut.mprint("Starting from Iteration %d" % niter)
sess.run(tset.fetch_op, feed_dict=tset.fdict())

while niter < MAXITER and not ut.stop:

    # Validate model every so often
    if niter % VALFREQ == 0 and niter != 0:
        ut.mprint("Validating model")
        val_iter = vset.ndata // BSZ
        vset.niter = 0
        sess.run(vset.fetch_op, feed_dict=vset.fdict())

        vavg_loss, vmin_loss, v_nvalids = 0., 0., 0.
        for its in range(val_iter):

            # Run DORN to get features
            for bi in range(N_DORN_BATCH):
                sess.run(swpV)
                sess.run(
                    [prepare_op, vset.fetch_op],
                    feed_dict={biter: bi, **vset.fdict()})

            # Get multiple pred with different noise
            vloss = []
            for viter in range(VARIter):
                sess.run(l2_op, feed_dict={l2_iter: viter})
            outs = sess.run([avg_l2, min_l2, l2_nvalids])
            vavg_loss += outs[0]
            vmin_loss += outs[1]
            v_nvalids += outs[2]

        vavg_loss, vmin_loss = vavg_loss / v_nvalids, vmin_loss / v_nvalids
        ut.vprint(niter, ['l2_min.v', 'l2_avg.v'], [vmin_loss, vavg_loss])

    # Run DORN to get features
    for bi in range(N_DORN_BATCH):
        sess.run(swpT)
        sess.run(
            [prepare_op, tset.fetch_op], feed_dict={biter: bi, **tset.fdict()})

    outs, _ = sess.run([lvals, tstep], feed_dict={lr: LR})
    ut.vprint(niter, tnms, outs)

    # Val on training set every so often
    if niter % VALFREQ == 0:

        tavg_loss, tmin_loss, t_nvalids = 0., 0., 0.
        for viter in range(VARIter):
            sess.run(l2_op, feed_dict={l2_iter: viter})
        outs = sess.run([avg_l2, min_l2, l2_nvalids])
        tavg, tmin = outs[0] / outs[2], outs[1] / outs[2]
        ut.vprint(niter, ['l2_min.t', 'l2_avg.t'], [tmin, tavg])

    niter = niter + 1

    # Save model weights if needed
    if SAVEFREQ > 0 and niter % SAVEFREQ == 0:
        mfn = WTS + "/iter_%06d.model.npz" % niter
        sfn = WTS + "/iter_%06d.state.npz" % niter

        ut.mprint("Saving model to " + mfn)
        ut.save_net(mfn, VAE_model.weights, sess)
        ut.mprint("Saving state to " + sfn)
        ut.save_adam(sfn, opt, VAE_model.weights, sess)
        ut.mprint("Done!")
        msave.clean(every=SAVEFREQ, last=1)
        ssave.clean(every=SAVEFREQ, last=1)

# Save last
if msave.iter < niter:
    mfn = WTS + "/iter_%06d.model.npz" % niter
    sfn = WTS + "/iter_%06d.state.npz" % niter

    ut.mprint("Saving model to " + mfn)
    ut.save_net(mfn, VAE_model.weights, sess)
    ut.mprint("Saving state to " + sfn)
    ut.save_adam(sfn, opt, VAE_model.weights, sess)
    ut.mprint("Done!")
    msave.clean(every=SAVEFREQ, last=1)
    ssave.clean(every=SAVEFREQ, last=1)
