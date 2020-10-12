#!/usr/bin/env python3

from collections import Counter

import numpy as np
import tensorflow as tf

from prdepth import sampler
import prdepth.utils as ut


TLIST = 'data/test.txt'
THRESHOLD = 1.02
MEAN_THRESHOLD = 1.03

H, W = sampler.H, sampler.W
IH, IW = sampler.IH, sampler.IW
PSZ = sampler.PSZ
STRIDE = sampler.STRIDE
HNPS, WNPS = sampler.HNPS, sampler.WNPS


def ceil_up(x):
    if x - np.ceil(x) == 0:
        return int(x) + 1
    else:
        return int(np.ceil(x))


def get_patch_indices(y, x):
    ''' Get the indices of all patches containing the pixel in given location.

    Args:
        y: y loc of the pixel.
        x: x loc of the pixel.
    Returns:
        a list of patch indices each of which is (y, x), representing the
        (y-th, x-th) patch of all extracted patches.
    '''
    y_low = ceil_up((y - PSZ) / float(STRIDE))
    y_low = np.maximum(0, y_low)

    y_high = int(np.floor(y / float(STRIDE)))
    y_high = np.minimum(y_high, HNPS)

    x_low = ceil_up((x - PSZ) / float(STRIDE))
    x_low = np.maximum(0, x_low)

    x_high = int(np.floor(x / float(STRIDE)))
    x_high = np.minimum(x_high, WNPS)

    patch_indices = [(i, j) for i in range(y_low, y_high)
                     for j in range(x_low, x_high)]
    return patch_indices


def find_patch(loc_1, loc_2):
    ''' Find all patches that contains both pixels in the given locations.

    Args:
        loc_1: (y, x) loc of the first pixel.
        loc_2: (y, x) loc of the second pixel.
    Returns:
        patch_index: an 1-d array of patch indices containing both pixels. Each
            index is the index of the patch in the flattened output of 
            ut.PO.extract_patches (of shape (HNPS * WNPS, )).
        rel_index_1: the relative index of the first pixel in each of the
            found (flattened) patch (of shape (PSZ * PSZ, )).
        rel_index_2: the relative index of the second pixel in each of the
            found (flattened) patch (of shape (PSZ * PSZ, )).

        It returns None when there is no patch containing given two pixels.
    '''
    y1, x1 = loc_1
    y2, x2 = loc_2

    patch_indices_1 = get_patch_indices(y1, x1)
    patch_indices_2 = get_patch_indices(y2, x2)

    patch_indices = list(set(patch_indices_1) & set(patch_indices_2))
    if not patch_indices:
        return None, None, None
    patch_indices = np.array([list(ids) for ids in patch_indices])

    rel_index_1 = (y1 - patch_indices[:, 0] * STRIDE) * PSZ + \
        (x1 - patch_indices[:, 1] * STRIDE)
    rel_index_2 = (y2 - patch_indices[:, 0] * STRIDE) * PSZ + \
        (x2 - patch_indices[:, 1] * STRIDE)

    patch_index = patch_indices[:, 0] * WNPS + patch_indices[:, 1]

    return patch_index, rel_index_1, rel_index_2


#########################################################################
# Compare the depth of input two pixels in all patches that contain them and
# and in sample predictions of those patches.
depth_sampler = sampler.Sampler(nsamples=100, read_gt=True)

PO = ut.PatchOp(1, IH, IW, PSZ, STRIDE)
patched_mean = tf.reduce_mean(depth_sampler.patched_samples, axis=0)
image_mean = PO.group_patches(patched_mean)
image_mean = tf.image.resize_images(image_mean, [H, W], align_corners=True)

patched_samples = tf.clip_by_value(depth_sampler.patched_samples, 0.01, 1.)
patched_samples = tf.reshape(
    patched_samples, [depth_sampler.nsamples, HNPS * WNPS, PSZ * PSZ])

indices_ph_1 = tf.placeholder(
    shape=[depth_sampler.nsamples, None, 3], dtype=tf.int32)
indices_ph_2 = tf.placeholder(
    shape=[depth_sampler.nsamples, None, 3], dtype=tf.int32)

pixel_samples_1 = tf.gather_nd(patched_samples, indices_ph_1)
pixel_samples_2 = tf.gather_nd(patched_samples, indices_ph_2)

ratio = pixel_samples_1 / (pixel_samples_2 + 1e-8)
label = tf.cast(ratio > THRESHOLD, tf.float32) - \
    tf.cast(ratio < 1 / THRESHOLD, tf.float32)
label = tf.reshape(label, [-1])

sess = tf.Session()
depth_sampler.load_model(sess)

#########################################################################
# Main Loop

flist = [i.strip('\n') for i in open(TLIST).readlines()]
gt_labels, mean_labels, sample_labels = [], [], []
for filename in flist:
    # Run VAE to sample patch-wise predictions.
    depth_sampler.sample_predictions(filename, sess)

    pairs = np.loadtxt(filename + '_pairs.txt', dtype=np.int32)

    # This implementation could be made substantially faster by paralleizing
    # the computation for different pairs of points.
    for pair in pairs:
        y1, x1, y2, x2, gt_label = pair

        mean_prediction = sess.run(image_mean).squeeze()
        mean_ratio = mean_prediction[y1, x1] / (mean_prediction[y2, x2] + 1e-8)
        mean_label = float(mean_ratio > MEAN_THRESHOLD) - \
            float(mean_ratio < 1 / MEAN_THRESHOLD)

        # Map locations to the DORN resolution.
        y1 = int(y1 / (H - 1) * (IH - 1))
        x1 = int(x1 / (W - 1) * (IW - 1))
        y2 = int(y2 / (H - 1) * (IH - 1))
        x2 = int(x2 / (W - 1) * (IW - 1))

        patch_index, rel_index_1, rel_index_2 = find_patch((y1, x1), (y2, x2))
        if patch_index is None:
            # When no patch contains these two pixels at the same time, fall
            # back to use the mean prediction.
            sample_label = mean_label
        else:
            # Find all patches and samples containing these two pixels.
            # Count in how many of them, one is farther, closer or equally
            # closer than the other.
            patch_index = np.tile(patch_index, (depth_sampler.nsamples, 1))
            rel_index_1 = np.tile(rel_index_1, (depth_sampler.nsamples, 1))
            rel_index_2 = np.tile(rel_index_2, (depth_sampler.nsamples, 1))
            sample_index, _ = np.indices(
                (depth_sampler.nsamples, patch_index.shape[-1]))
            indices_1 = np.stack(
                [sample_index, patch_index, rel_index_1], axis=-1)
            indices_2 = np.stack(
                [sample_index, patch_index, rel_index_2], axis=-1)

            sample_label = sess.run(
                label,
                feed_dict={indices_ph_1: indices_1, indices_ph_2: indices_2})
            sample_label = Counter(sample_label.tolist()).most_common(1)[0][0]

        gt_labels.append(gt_label)
        mean_labels.append(mean_label)
        sample_labels.append(sample_label)

gt_labels = np.array(gt_labels)
mean_labels = np.array(mean_labels)
sample_labels = np.array(sample_labels)

mean_acc = np.mean(gt_labels == mean_labels)
sample_acc = np.mean(gt_labels == sample_labels)
print('WKDR: Mean: %.2f%%, Samples: %.2f%%' %
      (100 - 100 * mean_acc, 100 - 100 * sample_acc))
