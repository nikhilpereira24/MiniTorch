from minitorch.fast_ops import matrix_multiply
from .tensor_data import (
    MAX_DIMS,
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
)
import numpy as np


def tensor_map(fn):
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # # works backwards from out storage to out_tensor, broadcasted to in_tensor, and then finally get values in in_storage to map
        # for x in range(len(out)):  # iterate through each empty element in out storage
        #     out_tensor_index = np.array(
        #         out_shape
        #     )  # create a array to store one singular index in the out_tensor
        #     to_index(
        #         x, out_shape, out_tensor_index
        #     )  # convert the ordinal in out to a tensor index in out_tensor
        #     in_tensor_index = np.array(in_shape)  # create an empty in_tensor_index
        #     broadcast_index(
        #         out_tensor_index, out_shape, in_shape, in_tensor_index
        #     )  # get the index in the small in_tensor by passing in the index of in tensor, out_shape, in shape and empty list of size smaller tensor shape
        #     # convert the index of the small tensor to a index in in_storage
        #     posOutStorage = index_to_position(
        #         out_tensor_index, out_strides
        #     )  # gets the position in Out Storage
        #     posInStorage = index_to_position(
        #         in_tensor_index, in_strides
        #     )  # gets the position in In Storage
        #     out[posOutStorage] = fn(
        #         in_storage[int(posInStorage)]
        #     )  # now apply the function on the in_storage[index] and set the value to out using position in Out storage
        # assert False
        out_index = np.zeros(MAX_DIMS, np.int32)
        in_index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return _map


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Simple version::

        for i:
            for j:
                out[i, j] = fn(a[i, j])

    Broadcasted version (`a` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0])

    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
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
        # assert False
        out_index = np.zeros(MAX_DIMS, np.int32)
        a_index = np.zeros(MAX_DIMS, np.int32)
        b_index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])
        # for x in range(len(out)):  # iterate through out storage
        #     out_tensor_indexA = np.array(
        #         out_shape
        #     )  # create a array to store one singular index in out tensor
        #     to_index(
        #         x, out_shape, out_tensor_indexA
        #     )  # call to_index to to get index of out tensor, x is ordinal, out_shape is shape of out tensor, out_index array to be overwritten
        #     small_shape_index_a = np.array(
        #         a_shape
        #     )  # create a list to store the index in small index A (In tensor A)
        #     small_shape_index_b = np.array(
        #         b_shape
        #     )  # create a list to store the index in small index B (In tensor B)
        #     broadcast_index(
        #         out_tensor_indexA, out_shape, a_shape, small_shape_index_a
        #     )  # broadcast the index of out tensor into A tensor
        #     broadcast_index(
        #         out_tensor_indexA, out_shape, b_shape, small_shape_index_b
        #     )  # Broadcast for B
        #     # convert the indices of tensors to indices in storage
        #     posA = index_to_position(small_shape_index_a, a_strides)
        #     posB = index_to_position(small_shape_index_b, b_strides)
        #     outPos = index_to_position(out_tensor_indexA, out_strides)
        #     out[outPos] = fn(a_storage[int(posA)], b_storage[int(posB)])

    return _zip


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      out = fn_zip(a, b)

    Simple version ::

        for i:
            for j:
                out[i, j] = fn(a[i, j], b[i, j])

    Broadcasted version (`a` and `b` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0], b[0, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        # assert False
        out_index = np.zeros(MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[j])
        # for index in range(
        #     len(out)
        # ):  # iterate through the length of out which will be the indices we fill
        #     out_tensor_index = np.array(
        #         out_shape
        #     )  # making a blank array to store out tensor indices
        #     to_index(
        #         index, out_shape, out_tensor_index
        #     )  # converts the index in out storage to a tensor index in out tensor
        #     A_tensor_index = out_tensor_index  # temporary keep the out tensor index
        #     A_tensor_index[
        #         reduce_dim
        #     ] = 0  # set the temporary index at the dimension of reduction to 0, this gives you the starting point in the tensor to take the reduction from
        #     starting_val = a_storage[
        #         index_to_position(A_tensor_index, a_strides)
        #     ]  # Gets the starting float value from the A tensor
        #     # Sequentially Iterate the cubes in A tensor of this dimension by applying the reduce function on each sequential cube
        #     for dim in range(
        #         1, a_shape[reduce_dim]
        #     ):  # iterate through the dimension of the A tensor from 1 through the shape for the reduce dimension, basically goes one cube at a time through the dimension we take the reduction of
        #         a_index = out_tensor_index  # copy the out_tensor index to A tensor so that we can get
        #         a_index[
        #             reduce_dim
        #         ] = dim  # set the dimension to dim which at first will be 1, then 2... how every many in in the a_shape[reduce_dim]
        #         new_float = a_storage[
        #             index_to_position(a_index, a_strides)
        #         ]  # get this value from a_storage by jumping to the correct position in storage
        #         reduce_value = fn(
        #             starting_val, new_float
        #         )  # reduce the starting_value and current value, this will always be a two value reduction
        #         starting_val = reduce_value  # update the starting reduce value
        #     out[index] = fn(out[index], starting_val)

    return _reduce


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`TensorData` : new tensor
    """

    f = tensor_reduce(fn)

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


# def tensor_matrix_multiply(
#     out,
#     out_shape,
#     out_strides,
#     a_storage,
#     a_shape,
#     a_strides,
#     b_storage,
#     b_shape,
#     b_strides,
# ):
#     """
#     NUMBA tensor matrix multiply function.
#     Should work for any tensor shapes that broadcast as long as ::
#         assert a_shape[-1] == b_shape[-2]
#     Optimizations:
#         * Outer loop in parallel
#         * No index buffers or function calls
#         * Inner loop should have no global writes, 1 multiply.
#     Args:
#         out (array): storage for `out` tensor
#         out_shape (array): shape for `out` tensor
#         out_strides (array): strides for `out` tensor
#         a_storage (array): storage for `a` tensor
#         a_shape (array): shape for `a` tensor
#         a_strides (array): strides for `a` tensor
#         b_storage (array): storage for `b` tensor
#         b_shape (array): shape for `b` tensor
#         b_strides (array): strides for `b` tensor
#     Returns:
#         None : Fills in `out`
#     """
#     # The first dimension in matmul is a broadcasted batch. This is code to help you handle those.
#     # batch dimension makes it 3 dimensions
#     #assert False
#     a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
#     b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
#     # broadcasted batch
#     # (0,3,2) x (0, 2, 3)

#     if a_shape[-1] == b_shape[-2]:
#         inner_index = a_shape[-1]
#         for n in range(out_shape[0]):  # iterate through depth
#             for i in range(out_shape[1]):  # iterate through row
#                 for j in range(out_shape[2]):  # iterate through column
#                     # gets positions in storage for this n,i,j index
#                     out_stor_pos = (
#                         n * out_strides[0] + i * out_strides[1] + j * out_strides[2]
#                     )  # starting positions in out
#                     a_stor_pos = (
#                         n * a_batch_stride + i * a_strides[1]
#                     )  # starting positions in A storage
#                     b_stor_pos = (
#                         n * b_batch_stride + j * b_strides[2]
#                     )  # starting positions in B storage
#                     jumpA = a_strides[2]  # jump of A
#                     jumpB = b_strides[1]  # jump of B
#                     temp = 0
#                     for k in range(
#                         inner_index
#                     ):  # loop this many times to (reduction loop)
#                         temp += a_storage[a_stor_pos] * b_storage[b_stor_pos]
#                         a_stor_pos += jumpA  # update the position in storage by jump
#                         b_stor_pos += jumpB  # update the position in storage by jump
#                     out[out_stor_pos] = temp


# def matrix_multiply(a, b):
#     """
#     Batched tensor matrix multiply ::

#         for n:
#           for i:
#             for j:
#               for k:
#                 out[n, i, j] += a[n, i, k] * b[n, k, j]
#     Where n indicates an optional broadcasted batched dimension.

#     Should work for tensor shapes of 3 dims ::

#         assert a.shape[-1] == b.shape[-2]

#     Args:
#         a (:class:`Tensor`): tensor data a
#         b (:class:`Tensor`): tensor data b

#     Returns:
#         :class:`Tensor` : new tensor data
#     """

#     # Make these always be a 3 dimensional multiply
#     both_2d = 0
#     if len(a.shape) == 2:
#         a = a.contiguous().view(1, a.shape[0], a.shape[1])
#         both_2d += 1
#     if len(b.shape) == 2:
#         b = b.contiguous().view(1, b.shape[0], b.shape[1])
#         both_2d += 1
#     both_2d = both_2d == 2

#     ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
#     ls.append(a.shape[-2])
#     ls.append(b.shape[-1])
#     assert a.shape[-1] == b.shape[-2]
#     out = a.zeros(tuple(ls))

#     tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

#     # Undo 3d if we added it.
#     if both_2d:
#         out = out.view(out.shape[1], out.shape[2])
#     return out


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
