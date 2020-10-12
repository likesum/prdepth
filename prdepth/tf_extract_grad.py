# https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/ops/array_grad.py#L725

from math import ceil

from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

def _ExtractImagePatchesGrad(op, grad):
    ''' Gradient function of tf.extract_image_patches. '''

    batch_size, rows_in, cols_in, channels = [
        dim.value for dim in op.inputs[0].get_shape()]
    input_bhwc = array_ops.shape(op.inputs[0])
    batch_size = input_bhwc[0]
    channels = input_bhwc[3]

    _, rows_out, cols_out, _ = [dim.value for dim in op.outputs[0].get_shape()]
    _, ksize_r, ksize_c, _ = op.get_attr("ksizes")
    _, stride_r, stride_h, _ = op.get_attr("strides")
    _, rate_r, rate_c, _ = op.get_attr("rates")
    padding = op.get_attr("padding")

    ksize_r_eff = ksize_r + (ksize_r - 1) * (rate_r - 1)
    ksize_c_eff = ksize_c + (ksize_c - 1) * (rate_c - 1)

    if padding == b"SAME":
        rows_out = int(ceil(rows_in / stride_r))
        cols_out = int(ceil(cols_in / stride_h))
        pad_rows = ((rows_out - 1) * stride_r + ksize_r_eff - rows_in) // 2
        pad_cols = ((cols_out - 1) * stride_h + ksize_c_eff - cols_in) // 2

    elif padding == b"VALID":
        rows_out = int(ceil((rows_in - ksize_r_eff + 1) / stride_r))
        cols_out = int(ceil((cols_in - ksize_c_eff + 1) / stride_h))
        pad_rows = (rows_out - 1) * stride_r + ksize_r_eff - rows_in
        pad_cols = (cols_out - 1) * stride_h + ksize_c_eff - cols_in

    pad_rows, pad_cols = max(0, pad_rows), max(0, pad_cols)

    grad_expanded = array_ops.transpose(
        array_ops.reshape(
            grad, (batch_size, rows_out, cols_out, ksize_r, ksize_c, channels)),
        (1, 2, 3, 4, 0, 5))
    grad_flat = array_ops.reshape(grad_expanded, (-1, batch_size * channels))

    row_steps = range(0, rows_out * stride_r, stride_r)
    col_steps = range(0, cols_out * stride_h, stride_h)

    idx = []
    for i in range(rows_out):
        for j in range(cols_out):
            r_low, c_low = row_steps[i] - pad_rows, col_steps[j] - pad_cols
            r_high, c_high = r_low + ksize_r_eff, c_low + ksize_c_eff

            idx.extend([(r * (cols_in) + c, i * (cols_out * ksize_r * ksize_c) + j
                         * (ksize_r * ksize_c) + ri * (ksize_c) + ci)
                        for (ri, r) in enumerate(range(r_low, r_high, rate_r))
                        for (ci, c) in enumerate(range(c_low, c_high, rate_c))
                        if 0 <= r and r < rows_in and 0 <= c and c < cols_in])

    sp_shape = (rows_in * cols_in, rows_out * cols_out * ksize_r * ksize_c)

    sp_mat = sparse_tensor.SparseTensor(
        array_ops.constant(idx, dtype=ops.dtypes.int64),
        array_ops.ones((len(idx),), dtype=ops.dtypes.float32), sp_shape)

    jac = sparse_ops.sparse_tensor_dense_matmul(sp_mat, grad_flat)

    grad_out = array_ops.reshape(jac, (rows_in, cols_in, batch_size, channels))
    grad_out = array_ops.transpose(grad_out, (2, 0, 1, 3))

    return [grad_out]