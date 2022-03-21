import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from numba import njit, prange

# TIP: Use `SET NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.
# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.
    Optimizations:
        * Main loop in parallel
        * All indices use numpy buffers
        * When `out` and `in` are stride-aligned, avoid indexing
    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.
    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        if (
            len(out_strides) != len(in_strides)
            or (out_strides != in_strides).any()
            or (out_shape != in_shape).any()
        ):
            for i in prange(len(out)):
                out_index = np.zeros(MAX_DIMS, np.int32)
                in_index = np.zeros(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])
        else:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        # iter = len(out)
        # # optimization case when out and in are stride identical and the length of both are same then just directly map storage
        # if np.array_equal(out_strides, in_strides) and len(out) == len(
        #     in_storage
        # ):  # if the strides and lengths are equal then just directly map the values in instorage to out
        #     for i in prange(iter):
        #         out[i] = fn(in_storage[i])  # index out
        # else:
        #     for i in prange(iter):
        #         out_index = np.zeros(MAX_DIMS, np.int32)
        #         in_index = np.zeros(MAX_DIMS, np.int32)
        #         to_index(i, out_shape, out_index)
        #         broadcast_index(out_index, out_shape, in_shape, in_index)
        #         o = index_to_position(out_index, out_strides)
        #         j = index_to_position(in_index, in_strides)
        #         out[o] = fn(in_storage[j])

    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::
      fn_map = map(fn)
      fn_map(a, out)
      out
    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`
    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.
    Optimizations:
        * Main loop in parallel
        * All indices use numpy buffers
        * When `out`, `a`, `b` are stride-aligned, avoid indexing
    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.
    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        if (
            len(out_strides) != len(a_strides)
            or len(out_strides) != len(b_strides)
            or (out_strides != a_strides).any()
            or (out_strides != b_strides).any()
            or (out_shape != a_shape).any()
            or (out_shape != b_shape).any()
        ):
            for i in prange(len(out)):
                out_index = np.empty(MAX_DIMS, np.int32)
                a_index = np.empty(MAX_DIMS, np.int32)
                b_index = np.empty(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                k = index_to_position(b_index, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])
        else:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])

        # optimization case when strides are aligned and storage lengths are aligned we can directly zip through storage
        # if (
        #     np.array_equal(out_strides, a_strides)
        #     and np.array_equal(out_strides, b_strides)
        #     and np.array_equal(len(a_storage), len(b_storage))
        # ):
        #     for i in prange(len(out)):
        #         out[i] = fn(a_storage[i], b_storage[i])
        # else:
        #     for i in prange(len(out)):
        #         out_index = np.zeros(MAX_DIMS, np.int32)
        #         a_index = np.zeros(MAX_DIMS, np.int32)
        #         b_index = np.zeros(MAX_DIMS, np.int32)
        #         to_index(i, out_shape, out_index)
        #         o = index_to_position(out_index, out_strides)
        #         broadcast_index(out_index, out_shape, a_shape, a_index)
        #         j = index_to_position(a_index, a_strides)
        #         broadcast_index(out_index, out_shape, b_shape, b_index)
        #         k = index_to_position(b_index, b_strides)
        #         out[o] = fn(a_storage[j], b_storage[k])

    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function.
      fn_zip = zip(fn)
      c = fn_zip(a, b)
    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over
    Returns:
        :class:`Tensor` : new tensor data
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.
    Optimizations:
        * Main loop in parallel
        * All indices use numpy buffers
        * Inner-loop should not call any functions or write non-local variables
    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out
    Returns:
        None : Fills in `out`
    """

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        for i in prange(len(out)):
            out_index = np.empty(MAX_DIMS, np.int32)
            reduce_size = a_shape[reduce_dim]
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            accum = out[o]
            j = index_to_position(out_index, a_strides)
            step = a_strides[reduce_dim]
            for s in range(reduce_size):
                accum = fn(accum, a_storage[j])
                j += step
            out[o] = accum

    return njit(parallel=True)(_reduce)
    # for i in prange(len(out)):  # iterates through out storage
    #     out_index = np.zeros(MAX_DIMS, np.int32)
    #     reduce_size = a_shape[
    #         reduce_dim
    #     ]  # how many times we need to reduce along the a shapes reduce dim axis
    #     to_index(
    #         i, out_shape, out_index
    #     )  # converts an ordinal to index in the out tensor
    #     o = index_to_position(
    #         out_index, out_strides
    #     )  # stores the position that needs to be updated in out
    #     # inner for loop needs to jump through the storage based on strides and then do the reduction in serialization
    #     # fn can be called but not index_to_function you need to come up with your jump algorithm based on strides
    #     out_index[
    #         reduce_dim
    #     ] = 0  # set index of the reduce dim to 0 for example [1,0,3]
    #     start = index_to_position(
    #         out_index, a_strides
    #     )  # starting value in a_storage based on this out tensor index
    #     jumper = a_strides[reduce_dim]  # how you jump through storage
    #     temp = fn(out[o], a_storage[start])  # calculate the first value temporary
    #     for _ in range(
    #         reduce_size - 1
    #     ):  # iterate through the positions in a_storage reduce size - 1 times
    #         start += jumper  # calculate the first start value
    #         temp = fn(temp, a_storage[start])  # update temp
    #     out[i] = temp  # set it


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::
      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)
    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce
    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


@njit(parallel=True, fastmath=True)
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    NUMBA tensor matrix multiply function.
    Should work for any tensor shapes that broadcast as long as ::
        assert a_shape[-1] == b_shape[-2]
    Optimizations:
        * Outer loop in parallel
        * No index buffers or function calls
        * Inner loop should have no global writes, 1 multiply.
    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor
    Returns:
        None : Fills in `out`
    """
    # The first dimension in matmul is a broadcasted batch. This is code to help you handle those.
    # batch dimension makes it 3 dimensions
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    assert a_shape[-1] == b_shape[-2]

    for i1 in prange(out_shape[0]):
        for i2 in prange(out_shape[1]):
            for i3 in prange(out_shape[2]):
                a_inner = i1 * a_batch_stride + i2 * a_strides[1]
                b_inner = i1 * b_batch_stride + i3 * b_strides[2]
                acc = 0.0
                for _ in range(a_shape[2]):
                    acc += a_storage[a_inner] * b_storage[b_inner]
                    a_inner += a_strides[2]
                    b_inner += b_strides[1]
                out_position = (
                    i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]
                )
                out[out_position] = acc
    # return out
    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # # broadcasted batch
    # # (0,3,2) x (0, 2, 3)

    # if a_shape[-1] == b_shape[-2]:
    #     inner_index = a_shape[-1]
    #     for n in prange(out_shape[0]):  # iterate through depth
    #         for i in prange(out_shape[1]):  # iterate through row
    #             for j in prange(out_shape[2]):  # iterate through column
    #                 # gets positions in storage for this n,i,j index
    #                 out_stor_pos = (
    #                     n * out_strides[0] + i * out_strides[1] + j * out_strides[2]
    #                 )  # starting positions in out
    #                 a_stor_pos = (
    #                     n * a_batch_stride + i * a_strides[1]
    #                 )  # starting positions in A storage
    #                 b_stor_pos = (
    #                     n * b_batch_stride + j * b_strides[2]
    #                 )  # starting positions in B storage
    #                 jumpA = a_strides[2]  # jump of A
    #                 jumpB = b_strides[1]  # jump of B
    #                 temp = 0
    #                 for k in range(
    #                     inner_index
    #                 ):  # loop this many times to (reduction loop)
    #                     temp += a_storage[a_stor_pos] * b_storage[b_stor_pos]
    #                     a_stor_pos += jumpA  # update the position in storage by jump
    #                     b_stor_pos += jumpB  # update the position in storage by jump
    #                 out[out_stor_pos] = temp
    # return out


def matrix_multiply(a, b):
    """
    Batched tensor matrix multiply ::

        for n:
          for i:
            for j:
              for k:
                out[n, i, j] += a[n, i, k] * b[n, k, j]
    Where n indicates an optional broadcasted batched dimension.

    Should work for tensor shapes of 3 dims ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor data a
        b (:class:`Tensor`): tensor data b

    Returns:
        :class:`Tensor` : new tensor data
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
