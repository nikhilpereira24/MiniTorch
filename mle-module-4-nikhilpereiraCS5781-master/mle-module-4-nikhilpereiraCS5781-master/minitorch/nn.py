from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert (
        height % kh == 0
    )  # checks to see that the height of the input can be divided in equal chunks by kh of the weights
    assert (
        width % kw == 0
    )  # checks to see that the width of the input can be divided into equal chunks by kw of the weights
    # You need to generalize the above idea in 1D pooling to create a shape with two extra dimesions to reduce over:
    nh = (
        height // kh
    )  # calculate the new height based on the input tensor height divided by the kernel height (4/1) or (4/2)
    nw = (
        width // kw
    )  # calculate the new width based on the input tensor width divided by the kernel width (4/1) or (4/2)
    # make a 6 dimensional tensor with tiles and flip the 4th and 5th dimension with a permute (ensure its contiguous)
    new_t = (
        input.contiguous()
        .view(batch, channel, nh, kh, nw, kw)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    # view it as a 5 dimensional tensor
    new_t = new_t.view(batch, channel, nh, nw, kh * kw)
    # return the new tensor, new height and new width
    return new_t, nh, nw


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    t, nh, nw = tile(input, kernel)  # tile the tensor with kernel
    t2 = t.mean(4).view(
        batch, channel, nh, nw
    )  # reduce along the last dimension and view
    return t2


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim)

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        inp, dim = ctx.saved_values
        # First compute argmax
        # Only send gradient to argmax gradinput
        # Everything else is 0
        return argmax(inp, dim) * grad_output


max = Max.apply


def softmax(input, dim):
    """
    Compute the softmax as a tensor.
    .. math::

        z_i = frac{e^{x_i}}{sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    numerator = input.exp()  # calculate the exp
    denominator = numerator.sum(dim)  # denominator is the sum of the input.exp()
    return numerator / denominator  # return the division


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    f = input - max(input, dim)  # inner parentehses
    a = f.exp()  # exponent
    b = a.sum(dim)  # reduction
    c = b.log()  # log
    d = input - c - max(input, dim)  # x - log - m
    return d


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    t, nh, nw = tile(input, kernel)
    t2 = max(t, 4).view(
        batch, channel, nh, nw
    )  # reduce along the last dimension and view
    return t2


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with random positions dropped out
    """
    if ignore:  # if ignore
        return input  # return input
    else:
        return input * (
            rand(input.shape, backend=input.backend) > rate
        )  # multiply input by a random tensor where the values are greater than dropout rate
