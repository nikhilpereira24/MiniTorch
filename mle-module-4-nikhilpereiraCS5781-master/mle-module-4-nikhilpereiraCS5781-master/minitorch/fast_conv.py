from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function
from numba import njit, prange
import numpy as np


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


@njit(parallel=True)
def tensor_conv1d(
    out,
    out_shape,
    out_strides,
    out_size,
    input_storage,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # trial 2 with batch and with out channels
    for b in prange(batch_):
        for o in prange(out_channels_):
            for j in prange(out_width):
                final_value = 0.0  # reduction variable
                for c in prange(in_channels):  # for every row in in_channels
                    for w in prange(kw):  # iterate through the weight array
                        pos_w = (
                            o * s2[0] + c * s2[1] + w * s2[2]
                        )  # get the weight position [current_channel, row, col]*[weight strides]
                        input_index = np.zeros(3, np.int16)
                        if reverse is False:  # if forward pass
                            # find the position in the input using strides but add position the weight to create the window effect
                            pos_in = b * s1[0] + c * s1[1] + j * s1[2] + w * s1[2]
                            val = j + w
                        else:
                            # find the position in the input using strides but subtracts position the weight to create the window effect in reverse
                            # input_index = [b, c, j-w]
                            pos_in = b * s1[0] + c * s1[1] + j * s1[2] - w * s1[2]
                            val = j - w

                        input_index[0] = b
                        input_index[1] = c
                        input_index[2] = val
                        # only allow the positions within the inputs width
                        if input_index[2] < width and reverse is False:
                            final_value += input_storage[pos_in] * weight[pos_w]
                        # only allow the position greater than 0
                        elif input_index[2] >= 0 and reverse is True:
                            final_value += input_storage[pos_in] * weight[pos_w]
                # store final value to out
                out_pos = b * out_strides[0] + o * out_strides[1] + j * out_strides[2]
                out[out_pos] = final_value

    # # Basic Algorithm
    # # 1. slide along input of window size k
    # # 2. zip weight with part of the input
    # # 3. reduce zipped weight to 1 value
    # # 4. store in out
    # if reverse is False:  # Forward pass
    #     for b in prange(batch_):  # Iterate through batch of out
    #         for o in prange(
    #             out_channels_
    #         ):  # Iterate through rows of out (out_channels)
    #             for j in prange(
    #                 out_width
    #             ):  # Iterate through colunns of out (out_width)
    #                 final_value = 0.0  # reduction variable
    #                 for c in prange(in_channels):  # for every row in in_channels
    #                     for w in prange(kw):  # iterate through the weight array
    #                         pos_w = o * s2[0] + c * s2[1] + w * s2[2]
    #                         # find the position in the input using strides but add position the weight to create the window effect
    #                         pos_in = b * s1[0] + c * s1[1] + j * s1[2] + w * s1[2]
    #                         val = j + w
    #                         if (
    #                             val < width
    #                         ):  # check that the window out column + width of kernel is less than the width of the input shape
    #                             final_value += (
    #                                 input_storage[pos_in] * weight[pos_w]
    #                             )  # do zip and reduce
    #                 # store final value to out
    #                 out_pos = (
    #                     b * out_strides[0] + o * out_strides[1] + j * out_strides[2]
    #                 )
    #                 out[out_pos] = final_value
    #                 # print(out)
    # else:
    #     for b in prange(batch_):  # Iterate through batch of out
    #         for o in prange(
    #             out_channels_
    #         ):  # Iterate through rows of out (out_channels)
    #             for j in prange(
    #                 out_width
    #             ):  # Iterate through colunns of out (out_width)
    #                 final_value = 0.0  # reduction variable
    #                 for c in prange(in_channels):  # for every row in in_channels
    #                     for w in prange(kw):  # iterate through the weight array
    #                         pos_w = o * s2[0] + c * s2[1] + w * s2[2]
    #                         # find the position in the input using strides but subtract position the weight to create the window effect in reverse
    #                         pos_in = b * s1[0] + c * s1[1] + j * s1[2] - w * s1[2]
    #                         val = j - w
    #                         if (
    #                             val >= 0
    #                         ):  # check that the window out column - width of kernel is less than the width of the input shape
    #                             final_value += input_storage[pos_in] * weight[pos_w]
    #                 # store final value to out
    #                 out_pos = (
    #                     b * out_strides[0] + o * out_strides[1] + j * out_strides[2]
    #                 )
    #                 out[out_pos] = final_value


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


@njit(parallel=True, fastmath=True)
def tensor_conv2d(
    out,
    out_shape,
    out_strides,
    out_size,
    input_storage,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]
    # trial 2 with batch and with out channels
    for b in prange(batch_):  # iterate through out's batch
        for o in prange(out_channels_):  # iterate through out channels
            for i in prange(out_height):  # iterate through out height
                for j in prange(out_width):  # iterate through out width
                    final_value = 0.0  # reduction variable
                    for c in prange(in_channels):  # for every row in in_channels
                        for w in prange(
                            kw
                        ):  # iterate through the width of weight array
                            for h in prange(
                                kh
                            ):  # iterate through the height of weight array
                                pos_w = (
                                    o * s20 + c * s21 + h * s22 + w * s23
                                )  # get the weight position
                                input_index = np.zeros(4, np.int16)
                                if reverse is False:  # if forward pass
                                    # find the position in the input using strides but add position the weight to create the window effect
                                    pos_in = (
                                        b * s10
                                        + c * s11
                                        + i * s12
                                        + j * s13
                                        + h * s12
                                        + w * s13
                                    )
                                    val1 = i + h
                                    val2 = j + w
                                else:
                                    # find the position in the input using strides but subtracts position the weight to create the window effect in reverse
                                    # input_index = [b, c, j-w]
                                    pos_in = (
                                        b * s10
                                        + c * s11
                                        + i * s12
                                        + j * s13
                                        - h * s12
                                        - w * s13
                                    )
                                    val1 = i - h
                                    val2 = j - w

                                input_index[0] = b
                                input_index[1] = c
                                input_index[2] = val1
                                input_index[3] = val2
                                # only allow the positions within the inputs width
                                if (
                                    input_index[3] < height
                                    and input_index[2] < width
                                    and reverse is False
                                ):
                                    final_value += input_storage[pos_in] * weight[pos_w]
                                # only allow the position greater than 0
                                elif (
                                    input_index[2] >= 0
                                    and input_index[3] >= 0
                                    and reverse is True
                                ):
                                    final_value += input_storage[pos_in] * weight[pos_w]
                    # store final value to out
                    out_pos = (
                        b * out_strides[0]
                        + o * out_strides[1]
                        + i * out_strides[2]
                        + j * out_strides[3]
                    )
                    out[out_pos] = final_value
    # if reverse is False:
    #     for b in prange(batch_):
    #         for o in prange(out_channels_):
    #             for i in prange(out_height):
    #                 for j in prange(out_width):
    #                     final_value = 0.0  # reduction variable
    #                     for c in prange(in_channels):  # for every row in in_channels
    #                         for w in prange(
    #                             kw
    #                         ):  # iterate through the width of weight array
    #                             for h in prange(
    #                                 kh
    #                             ):  # iterate through the height of weight array
    #                                 pos_w = (
    #                                     o * s20 + c * s21 + h * s22 + w * s23
    #                                 )  # get the weight position
    #                                 # find the position in the input using strides but add position the weight to create the window effect
    #                                 pos_in = (
    #                                     b * s10
    #                                     + c * s11
    #                                     + i * s12
    #                                     + j * s13
    #                                     + h * s12
    #                                     + w * s13
    #                                 )
    #                                 val1 = i + h
    #                                 val2 = j + w
    #                                 # only allow the positions within the inputs width
    #                                 if val2 < height and val1 < width:
    #                                     final_value += (
    #                                         input_storage[pos_in] * weight[pos_w]
    #                                     )
    #                                 # only allow the position greater than 0
    #                     # store final value to out
    #                     out_pos = (
    #                         b * out_strides[0]
    #                         + o * out_strides[1]
    #                         + i * out_strides[2]
    #                         + j * out_strides[3]
    #                     )
    #                     out[out_pos] = final_value
    #                     # print(out)
    # else:
    #     for b in prange(batch_):
    #         for o in prange(out_channels_):
    #             for i in prange(out_height):
    #                 for j in prange(out_width):
    #                     final_value = 0.0  # reduction variable
    #                     for c in prange(in_channels):  # for every row in in_channels
    #                         for w in prange(
    #                             kw
    #                         ):  # iterate through the width of weight array
    #                             for h in prange(
    #                                 kh
    #                             ):  # iterate through the height of weight array
    #                                 pos_w = (
    #                                     o * s20 + c * s21 + h * s22 + w * s23
    #                                 )  # get the weight position
    #                                 # find the position in the input using strides but subtracts position the weight to create the window effect in reverse
    #                                 # input_index = [b, c, j-w]
    #                                 pos_in = (
    #                                     b * s10
    #                                     + c * s11
    #                                     + i * s12
    #                                     + j * s13
    #                                     - h * s12
    #                                     - w * s13
    #                                 )
    #                                 val1 = i - h
    #                                 val2 = j - w
    #                                 if val1 >= 0 and val2 >= 0:
    #                                     final_value += (
    #                                         input_storage[pos_in] * weight[pos_w]
    #                                     )
    #                     # store final value to out
    #                     out_pos = (
    #                         b * out_strides[0]
    #                         + o * out_strides[1]
    #                         + i * out_strides[2]
    #                         + j * out_strides[3]
    #                     )
    #                     out[out_pos] = final_value
    #                     # print(out)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
