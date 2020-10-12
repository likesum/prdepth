import numpy as np

# projection mask of NYUv2
PMASK = np.zeros([480, 640], dtype=np.float64)
PMASK[44:471, 40:601] = 1.0

# sorted names
METRIC_NAMES = [
    'RMSE',
    'Mean RMSE',
    'Mean Log10',
    'Abs Rel Diff',
    'Squa Rel Diff',
    'delta < 1.25',
    'delta < 1.25^2',
    'delta < 1.25^3',
]


def get_metrics(
        depths, preds, projection_mask=True, masks=None, rmse_only=False):
    '''
    Args:
    depths: a list of ground truth depth maps, of dtype np.float64 
        in range (0,1).
    preds: a list of predictions, of dtype np.float64, in range (0,1).
    projection_mask: if use the valid projection mask of NYUv2, 
        in which case, the depth map has size 480x640.

    Returns: 
    A dictionary of different metrics.
    '''

    # Check shape and dtype
    assert len(preds) == len(depths)
    for i in range(len(preds)):
        assert preds[i].dtype == np.float64
        assert depths[i].dtype == np.float64
        assert preds[i].shape == depths[i].shape

    preds = np.stack(preds, axis=0) * 10.0
    depths = np.stack(depths, axis=0) * 10.0
    results = {}

    # Masks
    if masks:
        masks = np.stack(masks, axis=0)
        masks = np.float64(depths > 0) * np.float64(masks)
    else:
        masks = np.float64(depths > 0)
    if projection_mask:
        assert masks.shape[1:] == (480, 640)
        masks = masks * PMASK[None]

    masks = masks.astype(np.bool)
    npixels = np.sum(masks, axis=(1, 2))
    diff = preds - depths

    # MSE, RMSE
    mse = np.sum(diff**2.0 * masks, axis=(1, 2)) / npixels
    mse = np.mean(mse)
    rmse = np.sqrt(mse)
    if rmse_only:
        return rmse

    results['MSE'] = mse
    results['RMSE'] = rmse

    # Delta
    delta = np.maximum(preds / depths, depths / preds)
    delta1 = np.sum(np.float64(delta < 1.25) * masks, axis=(1, 2)) / npixels
    delta2 = np.sum(np.float64(delta < 1.25**2) * masks, axis=(1, 2)) / npixels
    delta3 = np.sum(np.float64(delta < 1.25**3) * masks, axis=(1, 2)) / npixels
    results['delta < 1.25'] = np.mean(delta1)
    results['delta < 1.25^2'] = np.mean(delta2)
    results['delta < 1.25^3'] = np.mean(delta3)

    # Absolute relative difference
    abrdiff = np.abs(diff) * masks / depths
    abrdiff = np.sum(abrdiff, axis=(1, 2)) / npixels
    results['Abs Rel Diff'] = np.mean(abrdiff)

    # Squared relative difference
    sqrdiff = np.square(diff) * masks / depths
    sqrdiff = np.sum(sqrdiff, axis=(1, 2)) / npixels
    results['Squa Rel Diff'] = np.mean(sqrdiff)

    # Mean log10
    log10 = np.abs(np.log10(preds) - np.log10(depths))
    log10 = np.sum(log10 * masks, axis=(1, 2)) / npixels
    results['Mean Log10'] = np.mean(log10)

    # Mean RMSE
    mrmse = np.sum(np.square(diff) * masks, axis=(1, 2)) / npixels
    mrmse = np.mean(np.sqrt(mrmse))
    results['Mean RMSE'] = mrmse

    return results
