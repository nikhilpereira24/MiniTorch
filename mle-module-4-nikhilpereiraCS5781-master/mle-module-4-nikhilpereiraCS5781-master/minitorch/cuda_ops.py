from numba import cuda
import numba
from .tensor_data import (
    to_index,
    index_to_position,
    TensorData,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


def tensor_map(fn):
    """
    CUDA higher-order tensor map function. ::
    fn_map = tensor_map(fn)
    fn_map(out, ... )
    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        out_size (array): size for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.
    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        i = THREADS_PER_BLOCK * cuda.blockIdx.x + cuda.threadIdx.x
        # finds the address of the thread
        if i < out_size:  # checks the threads needed
            # does the regular map steps to every position in tensor storage
            out_index = cuda.local.array(MAX_DIMS, numba.int16)
            in_index = cuda.local.array(MAX_DIMS, numba.int16)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # o = index_to_position(in_index, in_strides)
            # j = index_to_position(in_index, in_strides)
            out[index_to_position(out_index, out_strides)] = fn(
                in_storage[index_to_position(in_index, in_strides)]
            )

    return cuda.jit()(_map)


def map(fn):
    # CUDA compile your kernel
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        # Instantiate and run the cuda kernel.
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    CUDA higher-order tensor zipWith (or map2) function ::
      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)
    Args:
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
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
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        i = THREADS_PER_BLOCK * cuda.blockIdx.x + cuda.threadIdx.x
        if i < out_size:  # checks that the threds for zip are in bounds
            # does the regular zip operations
            out_index = cuda.local.array(MAX_DIMS, numba.int16)
            a_index = cuda.local.array(MAX_DIMS, numba.int16)
            b_index = cuda.local.array(MAX_DIMS, numba.int16)
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)


def zip(fn):
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def _sum_practice(out, a, size):
    """
    This is a practice sum kernel to prepare for reduce.
    Given an array of length :math:`n` and out of size :math:`n // blockDIM`
    it should sum up each blockDim values into an out cell.
    [a_1, a_2, ..., a_100]
    |
    [a_1 +...+ a_32, a_32 + ... + a_64, ... ,]
    Note: Each block must do the sum using shared memory!
    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        size (int):  length of a.
    """
    BLOCK_DIM = 32
    shared = numba.cuda.shared.array(
        BLOCK_DIM, numba.float64
    )  # create the shared storage
    i = (
        numba.cuda.blockIdx.x * THREADS_PER_BLOCK + numba.cuda.threadIdx.x
    )  # find the address of the thread
    localID = numba.cuda.threadIdx.x
    blockID = numba.cuda.blockIdx.x
    if (
        i < size
    ):  # if the threads are in bounds of the array size update shared with the data
        shared[localID] = a[i]
    else:  # pad the shared memory with 0's for threads outside shared data
        shared[localID] = 0

    thread_filter = 1  # initial thread filter
    while thread_filter < BLOCK_DIM:  # while 1 < 32
        numba.cuda.syncthreads()  # sync the threads
        if (
            localID % (thread_filter * 2) == 0
        ):  # activate the right number of threads so every 2 threads on first pass, then 4 threads
            shared[localID] += shared[
                localID + thread_filter
            ]  # add the values thread_filter apart of activated threads
        thread_filter *= 2  # update the counter
    out[blockID] = shared[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a):
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.
    Args:
        fn: reduction function maps two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out
    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_dim,
        reduce_value,
    ):
        BLOCK_DIM = 1024
        shared = numba.cuda.shared.array(
            BLOCK_DIM, numba.float64
        )  # create the shared storage
        localID = numba.cuda.threadIdx.x

        if localID < a_shape[reduce_dim]:
            blockID = numba.cuda.blockIdx.x
            out_index = numba.cuda.local.array(
                MAX_DIMS, numba.int16
            )  # create a local array of tensor indices
            to_index(
                blockID, out_shape, out_index
            )  # find the index in out tensor based on out shape and block id (each block id is a reduce column)
            out_position = index_to_position(
                out_index, out_strides
            )  # just finds it in out_poistion for later mapping
            A_start = index_to_position(
                out_index, a_strides
            )  # finds the out tensor index in a storage, this is your Starting position in A Storage
            # if the threads are in bounds of the a tensor - reduce dim (we only need reduce size threads per block)
            # calculate the positions for every ordinal (in parallel)
            shared[localID] = a_storage[
                A_start + localID * a_strides[reduce_dim]
            ]  # copy data to the shared block for each reduce dim array
        else:  # pad the shared memory with reduce values for threads outside shared data
            shared[localID] = reduce_value  # pad with 1s if multiply and 0 otherwise

        thread_filter = 1  # initial thread filter
        while (
            thread_filter < BLOCK_DIM
        ):  # while our thread filter is less than the block Dim
            numba.cuda.syncthreads()  # sync the threads
            if (
                localID % (thread_filter * 2) == 0
            ):  # activate the right number of threads so every 2 threads on first pass, then 4 threads
                shared[localID] = fn(
                    shared[localID], shared[localID + thread_filter]
                )  # add the values thread_filter apart of activated threads
            thread_filter *= 2  # update the thread filter
        numba.cuda.syncthreads()
        if localID == 0:  # only use the locals of 0
            out[out_position] = shared[0]

    return cuda.jit()(_reduce)


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
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce
    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_reduce(cuda.jit(device=True)(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
        out_a = a.zeros(tuple(out_shape))
        threadsperblock = 1024
        blockspergrid = out_a.size
        f[blockspergrid, threadsperblock](
            *out_a.tuple(), out_a.size, *a.tuple(), dim, start
        )
        return out_a

    return ret


def _mm_practice(out, a, b, size):
    """
    This is a practice square MM kernel to prepare for matmul.
    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].
    Size is always < 32.
    Requirements:
      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.
    Compute ::
    for i:
        for j:
             for k:
                 out[i, j] += a[i, k] * b[k, j]
    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        b (array): storage for `a` tensor.
        size (int): size of the square
    """
    # BLOCK_DIM = 32
    # Create all the Global and Local X,Y IDs
    # Define A, B Matrix in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)
    sB = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)
    x = numba.cuda.blockIdx.x * THREADS_PER_BLOCK + numba.cuda.threadIdx.x
    y = numba.cuda.blockIdx.y * THREADS_PER_BLOCK + numba.cuda.threadIdx.y
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # check the global threads are in bounds of the shape (this works because our A, B, C are same)
    if x < size and y < size:
        # Each thread computes one element in the result matrix.
        # The dot product is chunked into dot products of TPB-long vectors.
        sA[tx, ty] = a[x * size + y]
        sB[tx, ty] = b[x * size + y]
        # Wait until all threads finish preloading data
        numba.cuda.syncthreads()
    else:  # pad with 0s
        sA[tx, ty] = 0
        sB[tx, ty] = 0
    # Computes partial product on the shared memory
    t = 0
    for k in range(size):
        t += sA[tx, k] * sB[k, ty]
    # Wait until all threads finish computing
    out[x * size + y] = t  # write to out


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a, b):
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


@cuda.jit()
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    out_size,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    CUDA tensor matrix multiply function.
    Requirements:
      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.
    Should work for any tensor shapes that broadcast as long as ::
        assert a_shape[-1] == b_shape[-2]
    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor
    Returns:
        None : Fills in `out`
    """
    # need to avoid thread reading all data from a and b
    # use shareed memory to load all memory needed
    # my neighbor is gonna use all the same shared memory

    # First case, assume A and B are square and take same shape as C
    # Simplest case load all of A to one shared and B to one shared to shared memory and compute value
    # Do a for loop to actually compute matrix multiplication
    # This is shown in Basic Cuda - Square Small

    # only difference in indexing is assume 3 dimensions, block handles 1st dim, one thread for

    # when we scale it we cannot load all of A and all of B into shared memory

    # harder case - load in one quadrant in at a time
    # Each block is one chunk of C so need to load A0.0 and B0.0 then load A0.1, B1.0
    # once you reach end of shared memoery load the next chunks to finish it off
    # uses a fixed block size but doesnt care about size of matrices
    # single thread for every output position, but divided into blocks and does part of the matrix multiply

    # things to be careful about
    # blocks dont fully cover A and B, need to check boundary condition to load value else load 0
    # his code does the chunking for you

    # blocks are independent of each other - own loading, reading writing
    # complicated part is within a block - commnicating things with shared

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    BLOCK_DIM = 32

    # Shared Memory for A and B
    sA = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    sB = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Local X,Y thread ids to access shared memory
    localx = cuda.threadIdx.x
    localy = cuda.threadIdx.y

    blockX = numba.cuda.blockIdx.x  # Block id of X dimension
    blockY = numba.cuda.blockIdx.y  # Block id of Y dimension
    z = (
        numba.cuda.blockIdx.z
    )  # Just need Block Id in the depth dimension [0 for a 2d matrix, 1 for 1 depth, 5 for 5 depth]

    position_x = (
        blockX * BLOCK_DIM + localx
    )  # Length of Block X equals to the ID [0,1,2..]*[0,1,..32] + 32 = Position 0,1-32, 64... + the thread x, these are global positions
    position_y = (
        blockY * BLOCK_DIM + localy
    )  # Length of Block Y equals to the ID [0,1,2..]*32 = Position 0, 32, 64... + the thread y, these are global positions

    # Temporary Variable that accumulates the result to be written to out
    t = 0
    for s in range(
        (a_shape[-1] + (BLOCK_DIM - 1)) // BLOCK_DIM
    ):  # iterates by 1 for every block in multiples of 32, Inner index tells you how many blocks you need to do the matrix multiply

        # calculates a jump so that within the first block its set to 0, and the threads 32+ its set to 32 which is the first thread of the next block
        buffer_x = (
            s * BLOCK_DIM + localx
        )  # finds the position of the x thread based on block iteration s
        buffer_y = (
            s * BLOCK_DIM + localy
        )  # finds the position of the y thread based on block iteration s

        # Checks that the x threads are within the horizontal axis (<number of columns) for tensor A
        # Checks that the y threads are within the vertical axis (<number of rows) for tensor A
        if buffer_x < a_shape[-1] and position_y < a_shape[-2]:
            # calculates position in storage (index to position function written out)
            position_A = (
                a_batch_stride * z
                + buffer_x * a_strides[-1]
                + position_y * a_strides[-2]
            )
            sA[localy, localx] = a_storage[position_A]  # stores the value in A storage

        else:  # pads the remaining locations of sharedA with 0s
            sA[localy, localx] = 0

        # Checks that the x threads are within the horizontal axis (<number of columns) for tensor B
        # Checks that the y threads are within the vertical axis (<number of rows) for tensor B
        if position_x < b_shape[-1] and buffer_y < b_shape[-2]:
            # calculates position in storage B (index to position function written out)
            position_B = (
                b_batch_stride * z
                + position_x * b_strides[-1]
                + buffer_y * b_strides[-2]
            )
            sB[localy, localx] = b_storage[position_B]

        else:  # pads the remaining locations of shared with 0s
            sB[localy, localx] = 0

        numba.cuda.syncthreads()
        # Computes partial product on the shared memory
        for k in range(BLOCK_DIM):
            t += sA[localy, k] * sB[k, localx]
        numba.cuda.syncthreads()

    if (
        position_y < out_shape[-2] and position_x < out_shape[-1]
    ):  # if global positions are within bounds of the rows and columns
        out_position = (
            z * out_strides[0]
            + position_y * out_strides[-2]
            + position_x * out_strides[-1]
        )  # calculate the position
        out[out_position] = t  # write to out


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        class:`Tensor` : new tensor
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

    # One block per batch, extra rows, extra col
    blockspergrid = (
        (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,  # block x
        (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,  # block y
        out.shape[0],  # block z
    )
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

    tensor_matrix_multiply[blockspergrid, threadsperblock](
        *out.tuple(), out.size, *a.tuple(), *b.tuple()
    )

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class CudaOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
